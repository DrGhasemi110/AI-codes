classdef BANN
    properties
        M               % Number of networks
        networks        % Cell array of neural networks
        numNeurons      % Current neuron counts per network
        sigma           % Residual std deviation
        p = 0.4;        % Growth probability
        lambda_prior = 1; % Poisson prior mean
        lambda_reg = 0.001; % L2 regularization
        maxIter = 200;   % Max MCMC iterations
        activation = 'relu'; % Activation function
        nu = 3;         % For inverse gamma prior
        lambda_sigma    % Prior hyperparameter
        val_error       % Validation error history
        cv_metrics      % Cross-validation metrics storage
    end
    
    methods
        function obj = BANN(M, varargin)
            % Constructor
            obj.M = M;
            obj.networks = cell(M, 1);
            obj.numNeurons = ones(M, 1);
            obj.cv_metrics = struct();
            
            % Parse optional parameters
            for i = 1:2:length(varargin)
                switch lower(varargin{i})
                    case 'p', obj.p = varargin{i+1};
                    case 'lambda_prior', obj.lambda_prior = varargin{i+1};
                    case 'lambda_reg', obj.lambda_reg = varargin{i+1};
                    case 'maxiter', obj.maxIter = varargin{i+1};
                    case 'activation', obj.activation = varargin{i+1};
                    case 'nu', obj.nu = varargin{i+1};
                end
            end
        end
        
        function obj = fit(obj, X, y)
            % Split data (70% train, 30% validation)
            cv = cvpartition(length(y), 'Holdout', 0.3);
            X_train = X(cv.training,:);
            y_train = y(cv.training);
            X_val = X(cv.test,:);
            y_val = y(cv.test);
            
            % Initialize networks and residuals
            obj.lambda_sigma = var(y_train);
            obj.sigma = std(y_train);
            obj.val_error = zeros(obj.maxIter, 1);
            residual_train = y_train;
            residual_val = y_val;
            
            % Initialize each network with 1 neuron
            for k = 1:obj.M
                net = fitrnet(X_train, y_train/obj.M, 'LayerSizes', 1, ...
                    'Lambda', obj.lambda_reg, 'Standardize', true, ...
                    'Activations', obj.activation, 'IterationLimit', 100);
                obj.networks{k} = net;
                residual_train = residual_train - predict(net, X_train);
                residual_val = residual_val - predict(net, X_val);
            end
            
            % MCMC iterations
            min_val_error = inf;
            for iter = 1:obj.maxIter
                for k = 1:obj.M
                    % Compute residual for current network
                    res_train_k = residual_train + predict(obj.networks{k}, X_train);
                    res_val_k = residual_val + predict(obj.networks{k}, X_val);
                    
                    % Propose new architecture
                    current_neurons = obj.numNeurons(k);
                    if rand < obj.p
                        proposed_neurons = current_neurons + 1;
                        operation = 'add';
                    else
                        proposed_neurons = max(1, current_neurons - 1);
                        operation = 'remove';
                    end
                    
                    % Train proposed network
                    net_proposed = fitrnet(X_train, res_train_k, ...
                        'Standardize', true, ...
                        'LayerSizes', proposed_neurons, ...
                        'Lambda', obj.lambda_reg, ...
                        'Activations', obj.activation, ...
                        'IterationLimit', 100);
                    
                    % Calculate acceptance probability
                    [A, ~, ~] = ...
                        obj.acceptance_prob(net_proposed, obj.networks{k}, ...
                        res_val_k, X_val, current_neurons, proposed_neurons, operation);
                    
                    % Accept/reject
                    if rand < A
                        obj.networks{k} = net_proposed;
                        obj.numNeurons(k) = proposed_neurons;
                    end
                    
                    % Update residuals
                    pred_train = predict(obj.networks{k}, X_train);
                    residual_train = res_train_k - pred_train;
                    residual_val = res_val_k - predict(obj.networks{k}, X_val);
                end
                
                % Update sigma from inverse gamma
                SSE = sum(residual_train.^2);
                a = (obj.nu + numel(y_train)) / 2;
                b = (obj.nu*obj.lambda_sigma + SSE) / 2;
                obj.sigma = sqrt(b / gamrnd(a, 1));
                
                % Track validation error
                obj.val_error(iter) = mean(residual_val.^2);
                if obj.val_error(iter) < min_val_error
                    min_val_error = obj.val_error(iter);
                elseif iter > 20 && (obj.val_error(iter) - min_val_error) > 0.0001
                    break; % Early stopping
                end
            end
        end
        
        function y_pred = predict(obj, X)
            y_pred = zeros(size(X,1), 1);
            for k = 1:obj.M
                y_pred = y_pred + predict(obj.networks{k}, X);
            end
        end
        
        function [A, L_current, L_proposed] = acceptance_prob(...
                obj, net_proposed, net_current, residual_val, X_val, ...
                m_current, m_proposed, operation)
            % Calculate likelihoods
            pred_current = predict(net_current, X_val);
            pred_proposed = predict(net_proposed, X_val);
            err_current = residual_val - pred_current;
            err_proposed = residual_val - pred_proposed;
            
            L_current = exp(-0.5 * sum(err_current.^2) / obj.sigma^2);
            L_proposed = exp(-0.5 * sum(err_proposed.^2) / obj.sigma^2);
            
            % Calculate priors
            prior_current = poisspdf(m_current, obj.lambda_prior);
            prior_proposed = poisspdf(m_proposed, obj.lambda_prior);
            
            % Calculate transition probabilities
            if strcmp(operation, 'add')
                T_forward = obj.p;
                T_reverse = 1 - obj.p;
            else
                T_forward = 1 - obj.p;
                T_reverse = obj.p;
            end
            
            % Acceptance probability
            A = min(1, (T_reverse * L_proposed * prior_proposed) / ...
                (T_forward * L_current * prior_current));
        end
        
        function obj = cross_validate(obj, X, y, k, reps)
            % Perform repeated k-fold cross-validation
            n = size(X, 1);
            all_predictions = zeros(n, reps);
            
            parfor r = 1:reps
                % Create k-fold partition for this repetition
                cv = cvpartition(n, 'KFold', k);
                fold_pred = zeros(n, 1);
                
                for fold = 1:k
                    train_idx = training(cv, fold);
                    test_idx = test(cv, fold);
                    
                    % Train BANN on training fold
                    bann_fold = BANN(obj.M, ...
                        'lambda_prior', obj.lambda_prior, ...
                        'lambda_reg', obj.lambda_reg, ...
                        'p', obj.p, ...
                        'activation', obj.activation, ...
                        'maxIter', 100);  % Reduced iterations for CV efficiency
                    
                    bann_fold = bann_fold.fit(X(train_idx, :), y(train_idx));
                    
                    % Predict on test fold
                    fold_pred(test_idx) = bann_fold.predict(X(test_idx, :));
                end
                
                all_predictions(:, r) = fold_pred;
                % fold_assignments(:, r) = cv.test;
            end
            
            % Calculate metrics
            obj.cv_metrics.predictions = all_predictions;
            % obj.cv_metrics.fold_assignments = fold_assignments;
 
            obj = obj.compute_cv_metrics(y, X);
        end
        
        function obj = compute_cv_metrics(obj, y_true, X)
            % Compute CV metrics from stored predictions
            n = length(y_true);
            predictions = obj.cv_metrics.predictions;
            
            % Pointwise metrics
            mean_pred = mean(predictions, 2);
            bias = y_true - mean_pred;
            variance = var(predictions, 0, 2);  % Pointwise variance
            mse_pointwise = mean((y_true - predictions).^2, 2);
            for i = 1:size(predictions,2)
                llh = (-0.5)*MaxLogLiklihood(predictions(:,i),y_true);
                [~,~,icF] = aicbic(llh,size(X,2),n,Normalize=true);
                AIC(i) = icF.aic;
                BIC(i) = icF.bic;
                AICC(i) = icF.aicc;
                CAIC(i) = icF.caic;
                HQC(i) = icF.hqc;
            end
            ResIC = [mean(AIC), mean(BIC), mean(AICC), mean(CAIC), mean(HQC)];
            % Overall metrics
            total_mse = mean(mse_pointwise);
            avg_bias = mean(bias);
            bias_squared = mean(bias.^2);
            avg_variance = mean(variance);
            
            % Variance decomposition
            mse_var = var(mse_pointwise);
            bias_var = var(bias);
            
            % Store metrics
            obj.cv_metrics.y_true = y_true;
            obj.cv_metrics.total_mse = total_mse;
            obj.cv_metrics.avg_bias = avg_bias;
            obj.cv_metrics.bias_squared = bias_squared;
            obj.cv_metrics.avg_variance = avg_variance;
            obj.cv_metrics.mse_variance = mse_var;
            obj.cv_metrics.bias_variance = bias_var;
            obj.cv_metrics.pointwise_bias = bias;
            obj.cv_metrics.pointwise_variance = variance;
            obj.cv_metrics.pointwise_mse = mse_pointwise;
            obj.cv_metrics.ResIC = ResIC;
            
            % Decomposition validation
            obj.cv_metrics.decomposition = bias_squared + avg_variance;
        end
        
        function report_cv_metrics(obj)
            % Display cross-validation metrics
            if isempty(obj.cv_metrics)
                error('No CV metrics available. Run cross_validate first.');
            end
            
            fprintf('\nCross-Validation Metrics:\n');
            fprintf('---------------------------------\n');
            fprintf('Total MSE: %.4f\n', obj.cv_metrics.total_mse);
            fprintf('Average Bias: %.4f\n', obj.cv_metrics.avg_bias);
            fprintf('Bias²: %.4f\n', obj.cv_metrics.bias_squared);
            fprintf('Average Variance: %.4f\n', obj.cv_metrics.avg_variance);
            fprintf('MSE Variance: %.4f\n', obj.cv_metrics.mse_variance);
            fprintf('Bias Variance: %.4f\n', obj.cv_metrics.bias_variance);
            fprintf('Bias² + Variance: %.4f\n', obj.cv_metrics.decomposition);
            fprintf('Decomposition Error: %.2e\n', ...
                abs(obj.cv_metrics.total_mse - obj.cv_metrics.decomposition));
            
            % Plot bias-variance decomposition
            figure;           
            scatter(obj.cv_metrics.y_true, mean(obj.cv_metrics.predictions, 2));
            hold on;
            plot([min(obj.cv_metrics.y_true), max(obj.cv_metrics.y_true)], ...
                 [min(obj.cv_metrics.y_true), max(obj.cv_metrics.y_true)], '-r');
            title('True vs Predicted Values');
            xlabel('True Values');
            ylabel('Predicted Values');
            grid on;

        end
    end
end

