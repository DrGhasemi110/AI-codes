function [SelectedModel, Bias, Variance, YPred, yPred, model,ypred_total] = ...
    ComprehensiveModelSelection(xData,yData,rep,name,direction,ee)
close all
rng(0,"twister")
model{1} = @(X,Y,Fold) fitrlinear(X, Y,'Regularization', 'lasso',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName',...
    'expected-improvement-plus','Optimizer',"gridsearch",'Verbose', 0, 'KFold', Fold ,'Showplots',0,...
    "MaxObjectiveEvaluations",30,"UseParallel",1));
model{2} = @(X,Y,Fold) fitrlinear(X, Y,'Regularization', 'ridge',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName',...
    'expected-improvement-plus','Optimizer',"gridsearch",'Verbose', 0, 'KFold', Fold ,'Showplots',0,...
    "MaxObjectiveEvaluations",30,"UseParallel",1));
%%
% Modified SVM Regression with enhanced settings
% Set global random seed for full reproducibility
rng(0,"twister")
% Fixed seed for all random processes
% Create fixed cross-validation partition
% Calculate data-dependent hyperparameter bounds
yStd = std(yData);
validKernels = {'gaussian', 'linear','rbf'};  % Exclude polynomial if needed
% Define constrained hyperparameter space
params = [
    optimizableVariable('BoxConstraint', [1e-1, 1e3], 'Transform', 'log'), ...
    optimizableVariable('KernelScale', [1e-3, max(10*std(xData))], 'Transform', 'log'), ...
    optimizableVariable('Epsilon', [0.001*yStd, 0.1*yStd], 'Transform', 'log'), ...
    optimizableVariable('KernelFunction', validKernels, 'Type', 'categorical')
    ];
model{3} = @(X,Y,Fold) fitrsvm(X, Y, ...
    'Standardize', true, ...
    'CacheSize', 'maximal', ...
    'OptimizeHyperparameters', params, ...
    'HyperparameterOptimizationOptions', struct(...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'Optimizer', 'bayesopt', ...           % More efficient than gridsearch
    'Verbose', 0, 'KFold', Fold , ...
    'Showplots', false, ...
    'MaxObjectiveEvaluations', 30, ...    % Increased evaluations
    'UseParallel', true));                           % Seed for optimization
%%
% Set global random seed for full reproducibility
rng(0,"twister")
% Create fixed cross-validation partition
% Calculate data-dependent hyperparameter bounds
yStd = std(yData);
% validKernels = {'ardsquaredexponential', 'ardmatern32', 'ardexponential'}; % ARD kernels
% Define constrained hyperparameter space
params = [
    optimizableVariable('KernelFunction',{'squaredexponential','exponential',...
    'matern32','matern52','rationalquadratic','ardsquaredexponential','ardexponential',...
    'ardmatern32','ardmatern52','ardrationalquadratic'}, 'Type', 'categorical'),...
    optimizableVariable('Sigma', [0.001*yStd, 0.3*yStd], 'Transform', 'log'), ...
    optimizableVariable('BasisFunction', {'constant', 'linear','pureQuadratic'}, 'Type', 'categorical')
    ];

% Train GPR model with constrained parameters
model{4} = @(X,Y,Fold) fitrgp(X, Y, ...
    'Standardize', true, ...
    'ActiveSetMethod', 'entropy', ...
    'OptimizeHyperparameters', params, ...
    'HyperparameterOptimizationOptions', struct(...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'Optimizer', 'bayesopt', ...
    'Verbose', 0, 'KFold', Fold , ...
    'Showplots', false, ...
    'MaxObjectiveEvaluations', 30, ...     % Increased from 10 to 30
    'UseParallel', true));
% if e == 1
% model{5} = @(X,Y,Fold) fitrnet(X,Y,'standardize',1,"LayerSizes",[10, 10],"Activations",["relu","relu"]);
model{5} = @(X,Y,Fold) fitrnet(X,Y,'standardize',1,'LayerSizes',[randperm(20,size(zeros(1,randperm(3,1)),2))],...
    'OptimizeHyperparameters', {'Lambda'}, ...
    'HyperparameterOptimizationOptions', struct(...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'Optimizer', 'bayesopt', ...
    'Verbose', 0, 'KFold', Fold , ...
    'Showplots', false, ...
    'MaxObjectiveEvaluations', 30, ...     % Increased from 10 to 30
    'UseParallel', true));
% else
%     % model{5} = @(X,Y,Fold) fitrnet(X,Y,'standardize',1,"LayerSizes",[10, 10],"Activations",["tanh","tanh"]);
%     model{5} = @(X,Y,Fold) fitrnet(X,Y,'standardize',1,...
%         'OptimizeHyperparameters', 'auto', ...
%         'HyperparameterOptimizationOptions', struct(...
%         'AcquisitionFunctionName', 'expected-improvement-plus', ...
%         'Optimizer', 'bayesopt', ...
%         'Verbose', 0, 'KFold', Fold , ...
%         'Showplots', false, ...
%         'MaxObjectiveEvaluations', 30, ...     % Increased from 10 to 30
%         'UseParallel', true));
% end

modelName = {'lasso','ridge','svm','gp','shallow nn'};
toSaveName = modelName;
% rep = 5*Fold;
for i = 1:5
    % [bias(i), variance(i), mean_test_mse(i), R2_aggregated(i), ~, ...
    %     yPred{i},mean_test_aic(i), mean_test_bic(i), mean_test_aicc(i), mean_test_caic(i),...
    %     mean_test_hqc(i),mse_ave(i),trainedModel{i},total_predictions{i}] = biasVarianceDecomposition(xData, yData, ...
    %     model{i}, Fold, rep);
    tic
    if ee == 1
        nn = {toSaveName{i},name};
        [best_K(i), trainedModel{i}, L_EE_star(i),...
            bias(i), variance(i), mean_test_mse(i), best_y_pred_Total{i},yPred{i},...
            mean_test_aic(i), mean_test_bic(i), mean_test_aicc(i), ...
            mean_test_caic(i),mean_test_hqc(i), R2_aggregated(i)] = ...
            ckcv_regression(model{i}, xData, yData, 0.05, rep, direction, nn);
    else
        nReps = 10;
        [bias(i), variance(i), mean_test_mse(i), R2_aggregated(i), ~,yPred{i},...
            mean_test_aic(i), mean_test_bic(i), mean_test_aicc(i), ...
            mean_test_caic(i),mean_test_hqc(i),~,...
            trainedModel{i},best_y_pred_Total{i}] = ...
            biasVarianceDecomposition(xData, yData, model{i}, 5, rep);
    end
    toc
    figure(i)
    plot(yData,yPred{i},'o','MarkerEdgeColor',[0, 0, 0.5])
    hold on
    grid on
    plot(yData,yData,'LineWidth',2,'Color','r')
    xlim([min(yData),max(yData)])
    ylim([min(yData),max(yData)])
    IC = [mean_test_aic(i), mean_test_bic(i), mean_test_aicc(i), mean_test_caic(i),...
        mean_test_hqc(i)];
    MEAN_IC(i) = mean(IC);
    annotation(figure(i),'textbox',...
        [0.6 0.1 0.1 0.3],...
        'String',{modelName{i},['MSE = ',num2str(bias(i)^2+variance(i))],...
        ['Bias = ',num2str(bias(i))],...
        ['Variance = ',num2str(variance(i))],...
        ['Mean IC = ', num2str(MEAN_IC(i))]},...
        'EdgeColor',[1 1 1],...
        'FontSize',14,...
        'FontName','Times New Roman',...
        'FitBoxToText','on');
    xlabel('yData')
    ylabel('yPred')
    set(gca,'fontname','times','fontsize', 14) ;
    nn = {toSaveName{i},name};
    print(fullfile(direction, strcat(nn{1},nn{2})), '-dpng', '-r1200');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

i = length(modelName)+1;
figure(i)
res=[mean_test_aic',mean_test_bic',mean_test_aicc',mean_test_caic',mean_test_hqc'];
plot(1:size(res,1),res,'-o','MarkerEdgeColor','auto','MarkerFaceColor','auto')
title('Information Criteria')
grid on
x_labels = {'Lasso','Ridge','SVM','GPR','NN'};
xticks(1:length(x_labels));
xticklabels(x_labels);
legend('AIC','BIC','AICC','CAIC','HQC');
lgd = legend;
lgd.Color = 'none';
set(gca,'fontname','times','fontsize', 14) ;
nn = {'Information Criteria',name};
print(fullfile(direction, strcat(nn{1},nn{2})), '-dpng', '-r1200');

% figure(i+1)
% hold on
% grid on
% plot(mean_test_mse,'-o','LineWidth',2,'Color',[0,0,0.5],'MarkerEdgeColor',[0,0,0.5],'MarkerFaceColor',[0,0,0.5])
% title('MSE')
% hold on
% grid on
% plot(mse_ave,'-*','LineWidth',2)
% xticks(1:length(x_labels));
% xticklabels(x_labels);
% clear idx
% [~,idx] = min(mean_test_mse);
% nn = {'MSE',name};
% print(fullfile(direction, strcat(nn{1},nn{2})), '-dpng', '-r1200');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(i+2)
plot(bias,'-o','LineWidth',2,'Color',[0,0,0.5],'MarkerEdgeColor',[0,0,0.5],'MarkerFaceColor',[0,0,0.5])
hold on
grid on
yyaxis right
plot(variance,'-*','LineWidth',2)
xticks(1:length(x_labels));
xticklabels(x_labels);
lgd = legend;
lgd.Color = 'none';
set(gca,'fontname','times','fontsize', 14) ;
legend('bias','variance')
nn = {'Bias Variance TO',name};
print(fullfile(direction, strcat(nn{1},nn{2})), '-dpng', '-r1200');

figure
semilogy(bias.^2+variance,'-d','LineWidth',2,'color',[0,0.5,0],'MarkerEdgeColor',[0,0.5,0],'MarkerFaceColor',[0,0.5,0])
grid on
xticks(1:length(x_labels));
xticklabels(x_labels);
nn = {'MSE TO',name};
print(fullfile(direction, strcat(nn{1},nn{2})), '-dpng', '-r1200');

SStot = var(yData)*length(yData); % total sum of squares
for i = 1:length(modelName)
    yPred{i} = reshape(yPred{i},size(yData));
    SSres = sum((yData-yPred{i}).^2); % residual sum of squares
    R2(i) = 1 - SSres/SStot; % coefficient of determination
end
figure
plot(R2,'-o','LineWidth',2,'Color',[0,0,0.5],'MarkerEdgeColor',[0,0,0.5],'MarkerFaceColor',[0,0,0.5])
grid on
ylabel('adj R^2')
xticks(1:length(x_labels));
xticklabels(x_labels);
set(gca,'fontname','times','fontsize', 14) ;
nn = {'adjusted R squered',name};
print(fullfile(direction,strcat(nn{1},nn{2}) ), '-dpng', '-r1200');
R2(R2<0) = 0;

%%
lambda = 0.5; %penalty factor for normalized variance
a = (bias.^2+variance)./R2;
b = variance;
a = (a - min(a))/(max(a) - min(a));
b = (b - min(b))/(max(b) - min(b));
c = a + lambda*b;
%%
% (bias.^2+variance)./R2 + variance


figure
plot((bias.^2+2*variance)./R2,'-o','LineWidth',2,'Color',[0,0,0.5],'MarkerEdgeColor',[0,0,0.5],'MarkerFaceColor',[0,0,0.5])
grid on
ylabel('k-fold mse_m')
xticks(1:length(x_labels));
xticklabels(x_labels);
set(gca,'fontname','times','fontsize', 14) ;
nn = {'modified K-Fold MSE',name};
print(fullfile(direction,strcat(nn{1},nn{2}) ), '-dpng', '-r1200');
% idx = R2<0;
% bias = bias(~idx); variance = variance(~idx); mdl = mdl(~idx);
[~, IDX] = min((bias.^2+variance),[],"all");
if length(IDX)>1
    r = R2(IDX); [~,idx] = max(r);
    IDX = IDX(idx);
end

% model{4} = @(X,Y) fitrgp(X, Y, ...
%     'Standardize', true, ...
%     'OptimizeHyperparameters', 'all', ...
%     'HyperparameterOptimizationOptions', struct(...
%         'AcquisitionFunctionName', 'expected-improvement-plus', ...
%         'Optimizer', 'bayesopt', ...
%
%         'Verbose', 0, 'KFold', Fold , ...
%         'Showplots', false, ...
%         'MaxObjectiveEvaluations', 30, ...     % Increased from 10 to 30
%         'UseParallel', true));
mdl = trainedModel{IDX};
ypred_total = best_y_pred_Total{IDX};
for i = 1:length(mdl)
    ypred(i,:) = predict(mdl{i}, xData);
end
ypred = mean(ypred,1);
ctr = 0;
% while std(ypred) < 0.1 * std(yData)
%     fprintf('Model underfitting - trying manual parameter tuning\n');
%
%     ctr = ctr + 1;
%     if ctr>10
%         break
%     end
%     fprintf('Model underfitting - trying manual parameter tuning\n');
%     mdl = model{IDX}(xData,yData);
%     ypred = predict(mdl, xData);
% end
SelectedModel = mdl;
Bias = bias(IDX);
Variance = variance(IDX);
YPred = yPred{IDX};
figure
plot(yData,yData)
hold on
grid on
plot(yData,ypred,'o')
end

