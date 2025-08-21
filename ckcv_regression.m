function [best_K, best_models, L_EE_star,...
    best_bias, best_variance, best_MSE, best_y_pred_Total,best_y_pred,...
    best_aic, best_bic, best_aicc, best_caic, best_hqc, best_R2] = ...
    ckcv_regression(Model, X, y, delta, l_MC, direction, nn)
rng(0,"twister")
global ctr_global
ctr_global = ctr_global + 1;
% C-KCV for regression with adaptive K-selection
% Inputs:
%   X: n x d feature matrix
%   y: n x 1 response vector
%   param_set: struct array of hyperparameters to search
%   delta: confidence level (e.g., 0.05)
%   l_MC: number of Monte Carlo iterations
% Outputs:
%   best_K: optimal number of folds
%   best_param: optimal hyperparameter set
%   best_models: cell array of trained models for best (K, param)
%   L_EE_star: estimated generalization error bound
ctr = 0;
n = size(X, 1);
K_min = 3;
K_max = 10; %min(20, floor(n/3));  % Ensure minimum 3 samples per fold
K_set = K_min:K_max;

% Initialize outputs
best_K = [];
best_param = [];
best_models = {};
L_MS_star = inf;
L_EE_star = inf;

% Main optimization loop
for K = K_set
    fold_size = floor(n/K);
    n_MS = fold_size;
    n_EE = fold_size;
    n_TR = n - n_MS;
    L_MS_sum = 0;
    L_EE_sum = 0;
    model_cell = cell(1, l_MC);

    for i = 1:l_MC
        % indices = crossvalind('Kfold', n, K);
        % Random data partitioning
        idx = randperm(n);
        idx_TR = idx(1:n_TR);
        idx_MS = idx(n_TR+1:n_TR+n_MS);
        idx_EE = idx(n_TR+n_MS+1:end);

        % Train initial model
        f_TR = train_model(Model, X(idx_TR,:), y(idx_TR), K);

        % Validation phase (MS)
        [~, residuals_MS] = predict_model(f_TR, X(idx_MS,:), y(idx_MS));
        [L_MS_i, ~] = compute_error_bound(residuals_MS, delta);
        [y_pred_Total(:,i), residuals_Total(:,i)] = predict_model(f_TR, X, y);
        llh = (-0.5)*MaxLogLiklihood(y_pred_Total(:,i),y);
        [~,~,icF] = aicbic(llh,size(X,2),size(X,1),Normalize=true);
        AIC(i) = icF.aic;
        BIC(i) = icF.bic;
        AICC(i) = icF.aicc;
        CAIC(i) = icF.caic;
        HQC(i) = icF.hqc;
        % Retrain on expanded set
        f_MS = train_model(Model, X([idx_TR, idx_MS],:), y([idx_TR, idx_MS]), K);
        model_cell{i} = f_MS;

        % Error estimation (EE)
        [~, residuals_EE] = predict_model(f_MS, X(idx_EE,:), y(idx_EE));
        [L_EE_i, ~] = compute_error_bound(residuals_EE, delta);

        % Accumulate errors
        L_MS_sum = L_MS_sum + L_MS_i;
        L_EE_sum = L_EE_sum + L_EE_i;
    end
    ctr = ctr + 1;
    % Calculate per-sample statistics

    yPred_avg = mean(y_pred_Total, 2, 'omitnan'); % Mean prediction per sample
    sample_variance = var(y_pred_Total, 0, 2, 'omitnan'); % Per-sample variance

    % Compute metrics
    residuals = y_pred_Total - y;
    squared_errors = residuals.^2;

    % Bias-variance decomposition
    bias_sq = mean((yPred_avg - y).^2); % Squared bias
    variance = mean(sample_variance); % Average per-sample variance
    mse_bvt = bias_sq +  variance; %mean(squared_errors(:), 'omitnan'); % Overall MSE
    bias_sq = sqrt(bias_sq);
    mse_ave = mean((yPred_avg - y).^2);
    bias = bias_sq;
    MSE = mse_bvt;
    aic = mean(AIC);
    bic = mean(BIC);
    aicc = mean(AICC);
    caic = mean(CAIC);
    hqc = mean(HQC);
    % R² calculations
    ss_res_aggregated = sum((yPred_avg - y).^2);
    ss_tot = sum((y - mean(y)).^2);
    R2_aggregated = 1 - ss_res_aggregated / ss_tot;
    se = mean(std(squared_errors,[],2));
    t_critical = tinv(1 - delta, size(squared_errors,1)-1);  % One-tailed t-value
    avg_L_MS = mse_bvt+ t_critical * se; %L_MS_sum / l_MC;
    avg_L_EE = L_EE_sum / l_MC;
    aa(ctr) = avg_L_MS;
    figure(ctr_global)
    plot(K_set(1:ctr),aa,'-ob','LineWidth',2,'MarkerFaceColor','b')
    hold on
    grid on
    xlabel('K')
    ylabel('MSE upper bound')
    set(gca,'fontName','Times New Roman','fontSize',16)
    drawnow

    % Update best parameters
    if avg_L_MS < L_MS_star
        L_MS_star = avg_L_MS;
        L_EE_star = avg_L_EE;
        best_K = K;
        best_models = model_cell;
        best_bias = bias;
        best_variance = variance;
        best_MSE = MSE;
        best_y_pred = mean(y_pred_Total,2);
        best_y_pred_Total = y_pred_Total;
        best_aic = aic;
        best_bic = bic;
        best_aicc = aicc;
        best_caic = caic;
        best_hqc = hqc;
        best_R2 = R2_aggregated;
    end
end
print(fullfile(direction, ['kselection ',strcat(nn{1},nn{2})]), '-dpng', '-r1200');

hold off
end

function [y_pred, residuals] = predict_model(model, X, y_true)
% Model prediction with residual calculation
y_pred = predict(model, X);  % Use MATLAB's predict method
residuals = y_true - y_pred;
end

function [error_bound, se] = compute_error_bound(residuals, delta)
% Compute statistical error bound
squared_errors = residuals.^2;
n = numel(squared_errors);
empirical_mse = mean(squared_errors);
se = std(squared_errors) / sqrt(n);
t_critical = tinv(1 - delta, n-1);  % One-tailed t-value
error_bound = empirical_mse + t_critical * se;
end

% Example train_model function for ridge regression
function model = train_model(Model, X_train, y_train, Fold)
rng(0);  % Fixed seed for all random processes
% Create fixed cross-validation partition
% Calculate data-dependent hyperparameter bounds
model = Model(X_train, y_train,Fold);
end