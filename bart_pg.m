function [trees, sigma_samples, predictions] = bart_pg(X, Y, m, num_iter, C, hyper)
    % BART with Particle Gibbs Sampling
    % Inputs:
    %   X: N x D matrix of features
    %   Y: N x 1 vector of responses
    %   m: Number of trees
    %   num_iter: MCMC iterations
    %   C: Number of particles for PG
    %   hyper: Struct of hyperparameters (alpha_s, beta_s, k, nu, q)
    % Outputs:
    %   trees: Cell array of sampled trees
    %   sigma_samples: Samples of noise variance
    %   predictions: Final prediction
    % Preprocess Y: Scale to [-0.5, 0.5]
    Ymin = min(Y);
    Ymax = max(Y);
    Y = (Y - Ymin) / (Ymax - Ymin) - 0.5;
    [N, D] = size(X);
    
    % Set default hyperparameters if not provided
    if nargin < 6
        hyper = struct();
        hyper.alpha_s = 0.95;
        hyper.beta_s = 2;
        hyper.k = 2;
        hyper.nu = 3;
        hyper.q = 0.9;
    end
    
    % Compute overestimate of sigma and set prior
    sigma0 = std(Y);
    lambda = (sigma0^2 / hyper.nu) * chi2inv(1 - hyper.q, hyper.nu);
    rate0 = hyper.nu * lambda / 2;
    sigma_mu = 0.5 / (hyper.k * sqrt(m));
    
    % Initialize trees, predictions, and sigma
    trees = cell(m, 1);
    f = zeros(N, 1);
    sigma = sigma0;
    sigma_samples = zeros(num_iter, 1);
    
    % Initialize each tree as a root node
    for j = 1:m
        trees{j} = initialize_root_tree(1:N);
    end
    
    % MCMC iterations
    for iter = 1:num_iter
        for j = 1:m
            % Compute residual R for tree j
            f_j = predict_tree(trees{j}, X);
            R = Y - (f);
            
            % Sample tree j using Particle Gibbs
            trees{j} = pg_sampler_tree(trees{j}, R, X, sigma, sigma_mu, C, hyper.alpha_s, hyper.beta_s);
            
            % Sample leaf parameters for tree j
            trees{j} = sample_leaf_parameters(trees{j}, R, sigma, sigma_mu);
            
            % Update prediction f
            f_j = predict_tree(trees{j}, X);
            f = f + f_j;
        end
        
        % Sample sigma^2 from inverse gamma
        SSR = sum((Y - f).^2);
        shape_sigma = (hyper.nu + N) / 2;
        scale_sigma = (rate0 * 2 + SSR) / 2;
        sigma2 = 1 / gamrnd(shape_sigma, 1/scale_sigma);
        sigma = sqrt(sigma2);
        sigma_samples(iter) = sigma;
        
        % % Display progress
        % if mod(iter, 10) == 0
        %     fprintf('Iteration %d/%d, sigma = %.4f\n', iter, num_iter, sigma);
        % end
    end
    predictions = f * (Ymax - Ymin) + Ymin + (Ymax - Ymin)/2; % Rescale back
end

%% Particle Gibbs sampler for a single tree
function new_tree = pg_sampler_tree(old_tree, R, X, sigma, sigma_mu, C, alpha_s, beta_s)
    particles = cell(C, 1);
    logW = zeros(C, 1);   % Incremental weights
    logL = zeros(C, 1);   % Log-likelihood
    
    % Initialize particles
    particles{1} = old_tree;
    logL(1) = compute_tree_logL(old_tree, R, sigma, sigma_mu);
    logW(1) = 0;  % Reference particle weight doesn't change
    
    for c = 2:C
        tree = initialize_root_tree(1:length(R));
        logL(c) = compute_tree_logL(tree, R, sigma, sigma_mu);
        logW(c) = 0;  % Start with zero weight
        particles{c} = tree;
    end
    
    % Check eligibility of non-reference particles
    non_ref_particles = particles(2:C);
    eligible_flags = cellfun(@(p) ~isempty(p.eligible_queue), non_ref_particles);
    
    % Continue while any non-reference particle has eligible nodes
    while any(eligible_flags)
        for c = 2:C
            if isempty(particles{c}.eligible_queue)
                continue;
            end
            
            % Process the next eligible node
            node_id = particles{c}.eligible_queue{1};
            particles{c}.eligible_queue(1) = [];
            node = particles{c}.nodes(node_id);
            
            % Check for valid splits
            [has_split, split_dims] = get_valid_splits(X, node.data_indices);
            p_split = alpha_s / (1 + node.depth)^beta_s * has_split;
            
            if rand < p_split
                % Sample a valid split
                [split_dim, split_value] = sample_split_uniform(X, node.data_indices, split_dims);
                [left_idx, right_idx] = split_data(X, node.data_indices, split_dim, split_value);
                
                % Create child nodes
                left_id = length(particles{c}.nodes) + 1;
                right_id = left_id + 1;
                left_suff = [length(left_idx), sum(R(left_idx)), sum(R(left_idx).^2)];
                right_suff = [length(right_idx), sum(R(right_idx)), sum(R(right_idx).^2)];
                
                left_node = struct('id', left_id, 'parent', node_id, 'depth', node.depth+1, ...
                                  'is_leaf', true, 'data_indices', left_idx, 'suff_stat', left_suff, ...
                                  'mu', 0, 'split_dim', [], 'split_value', [], ...
                                  'left_child', [], 'right_child', []);
                
                right_node = struct('id', right_id, 'parent', node_id, 'depth', node.depth+1, ...
                                   'is_leaf', true, 'data_indices', right_idx, 'suff_stat', right_suff, ...
                                   'mu', 0, 'split_dim', [], 'split_value', [], ...
                                   'left_child', [], 'right_child', []);
                
                % Update current node to internal
                particles{c}.nodes(node_id).is_leaf = false;
                particles{c}.nodes(node_id).split_dim = split_dim;
                particles{c}.nodes(node_id).split_value = split_value;
                particles{c}.nodes(node_id).left_child = left_id;
                particles{c}.nodes(node_id).right_child = right_id;
                
                % Append new nodes
                particles{c}.nodes = [particles{c}.nodes, left_node, right_node];
                particles{c}.eligible_queue = [particles{c}.eligible_queue, left_id, right_id];
                
                % Update marginal likelihood
                old_ml = compute_log_ml(node.suff_stat, sigma, sigma_mu);
                new_ml = compute_log_ml(left_suff, sigma, sigma_mu) + ...
                         compute_log_ml(right_suff, sigma, sigma_mu);
                delta_logL = new_ml - old_ml;
                logL(c) = logL(c) + delta_logL;
                logW(c) = logW(c) + delta_logL;
            end
        end
        
        % Resample non-reference particles
        non_ref_idx = 2:C;
        weights = exp(logW(non_ref_idx) - max(logW(non_ref_idx)));
        weights = weights / sum(weights);
        new_indices = randsample(non_ref_idx, C-1, true, weights);
        particles(non_ref_idx) = particles(new_indices);
        logW(non_ref_idx) = 0;  % Reset weights after resampling
        
        % Update eligibility flags
        non_ref_particles = particles(2:C);
        eligible_flags = cellfun(@(p) ~isempty(p.eligible_queue), non_ref_particles);
    end
    
    % Sample final tree using likelihood weights
    weights = exp(logL - max(logL));
    weights = weights / sum(weights);
    new_tree = particles{randsample(C, 1, true, weights)};
end

%% Helper functions
function tree = initialize_root_tree(data_indices)
    tree.nodes = struct('id', 1, 'parent', [], 'depth', 0, 'is_leaf', true, ...
                       'data_indices', data_indices, 'suff_stat', [0,0,0], ... % Placeholder
                       'mu', 0, 'split_dim', [], 'split_value', [], ...
                       'left_child', [], 'right_child', []);
    tree.eligible_queue = {1};  % Queue of node IDs to expand
end

function log_ml = compute_log_ml(suff_stat, sigma, sigma_mu)
    n = suff_stat(1);
    if n == 0
        log_ml = 0;
        return;
    end
    s = suff_stat(2);
    s2 = suff_stat(3);
    
    % Precompute terms for efficiency
    inv_sigma2 = 1/sigma^2;
    inv_sigma_mu2 = 1/sigma_mu^2;
    n_sigma_mu2 = n * sigma_mu^2;
    
    term1 = -n/2 * log(2*pi) - n/2 * log(sigma^2);
    term2 = -0.5 * log(1 + n_sigma_mu2 * inv_sigma2);
    term3 = -0.5 * inv_sigma2 * s2;
    term4 = 0.5 * (s^2 * inv_sigma2) * (inv_sigma_mu2 * sigma^2 / (sigma^2 + n_sigma_mu2));
    
    log_ml = term1 + term2 + term3 + term4;
end

function tree_logL = compute_tree_logL(tree, R, sigma, sigma_mu)
    tree_logL = 0;
    for i = 1:numel(tree.nodes)
        if tree.nodes(i).is_leaf
            suff_stat = tree.nodes(i).suff_stat;
            tree_logL = tree_logL + compute_log_ml(suff_stat, sigma, sigma_mu);
        end
    end
end

function [has_split, split_dims] = get_valid_splits(X, data_indices)
    split_dims = [];
    D = size(X, 2);
    for d = 1:D
        x_vals = X(data_indices, d);
        if numel(unique(x_vals)) >= 2
            split_dims = [split_dims, d];
        end
    end
    has_split = ~isempty(split_dims);
end

function [split_dim, split_value] = sample_split_uniform(X, data_indices, split_dims)
    d = split_dims(randi(numel(split_dims)));
    x_vals = unique(X(data_indices, d));
    split_points = (x_vals(1:end-1) + x_vals(2:end)) / 2;
    split_value = split_points(randi(numel(split_points)));
    split_dim = d;
end

function [left_idx, right_idx] = split_data(X, data_indices, split_dim, split_value)
    x_vals = X(data_indices, split_dim);
    left_mask = x_vals <= split_value;
    left_idx = data_indices(left_mask);
    right_idx = data_indices(~left_mask);
end

function tree = sample_leaf_parameters(tree, R, sigma, sigma_mu)
    for i = 1:numel(tree.nodes)
        if tree.nodes(i).is_leaf
            idx = tree.nodes(i).data_indices;
            n = numel(idx);
            if n == 0
                tree.nodes(i).mu = 0;
                continue;
            end
            s = sum(R(idx));
            
            % Posterior parameters
            prec_mu = 1/sigma_mu^2;
            prec_data = n/sigma^2;
            post_prec = prec_mu + prec_data;
            post_mean = (s/sigma^2) / post_prec;
            
            tree.nodes(i).mu = post_mean + randn/sqrt(post_prec);
        end
    end
end

function y_pred = predict_tree(tree, X)
    y_pred = zeros(size(X,1), 1);
    for i = 1:size(X,1)
        node_id = 1;
        while true
            node = tree.nodes(node_id);
            if node.is_leaf
                y_pred(i) = node.mu;
                break;
            else
                if X(i, node.split_dim) <= node.split_value
                    node_id = node.left_child;
                else
                    node_id = node.right_child;
                end
            end
        end
    end
end