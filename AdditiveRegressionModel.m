function [modelM, ResIC, mseM, R2M, bias, VarianceM, yLog, yLogSTD]...
    = AdditiveRegressionModel(xData,Y,Nmax,rep,name,direction,LogData)
close all
rng(0,"twister")
for i = 1:length(LogData)
    yLog{i} = zeros(size(LogData{i},1),1);
end
nn = {name,num2str(1)};
[modelM{1}, ~, ~, YPred, ~, ~,ypred_total{1}] = ...
    ComprehensiveModelSelection(xData,Y,rep,strcat(nn{1},nn{2}),direction,1);
% yPred = predict(model{1},xData);
yp = 0;
ypt = 0;
yp = yp + YPred;
ypt = ypt + ypred_total{1};
VarianceM(1) = mean(var(ypt,[],2));
SStot = var(Y)*length(Y); % total sum of squares
SSres = sum((Y-yp).^2); % residual sum of squares
R2M(1) = 1 - SSres/SStot; % coefficient of determination
error = Y - YPred;

E{1} = error;
% error = 2 * (error - min(error))./(max(error) - min(error)) - 1;
n = 1;
llh = (-0.5)*MaxLogLiklihood(yp,Y);
[~,~,icF] = aicbic(llh,size(xData,2),size(Y,1),Normalize=true);
aicM(n) = icF.aic;
bicM(n) = icF.bic;
aiccM(n) = icF.aicc;
caicM(n) = icF.caic;
hqcM(n) = icF.hqc;
mseM(n) = sqrt(mean((Y-yp).^2));
for j = 1:length(LogData)
    mdl = modelM{n};
    clear yy stdY
    for i = 1:length(mdl)
        if strcmp(class(mdl{i}),'RegressionGP')
            [yy(i,:),stdY(i,:)] = predict(mdl{i},LogData{j}(:,2:end)); %(predict(mdl,LogData{j}) + 1)/2*(MAX - MIN) + MIN;
        else
            yy(i,:) = predict(mdl{i},LogData{j}(:,2:end)); %(predict(mdl,LogData{j}) + 1)/2*(MAX - MIN) + MIN;
            stdY(i,1:size(yy,2)) = 0;
        end
    end
    yy = mean(yy,1);
    yLog{j} = yLog{j} + yy';
    yLogSTD{j} = mean(stdY',2);
end
rr = R2M(1);
bias(n) = sqrt(mean((Y-yp).^2));
while rr<.98
    n = n + 1;

    if n>Nmax
        n = n - 1;
        break
    end
    nn = {name,num2str(n)};
    [modelM{n}, ~, ~, YPred, ~, ~,ypred_total{n}] = ...
        ComprehensiveModelSelection(xData,error,rep,strcat(nn{1},nn{2}),direction,0);
    yp1 = yp;
    yp = yp + YPred;
    yp1 = yp1 + ypred_total{n};
    VarianceM(n) = mean(var(yp1,[],2));
    error = Y - yp;
    E{n} = error;
    % error = 2 * (error - min(error))./(max(error) - min(error)) - 1;
    SStot = var(Y)*length(Y); % total sum of squares
    SSres = sum((Y-yp).^2); % residual sum of squares
    R2M(n) = 1 - SSres/SStot; % coefficient of determination
    bias(n) = sqrt(mean((Y-yp).^2));
    mseM(n) = sqrt(mean((Y-yp).^2));
    llh = (-0.5)*MaxLogLiklihood(yp,Y);
    [~,~,icF] = aicbic(llh,size(xData,2),size(Y,1),Normalize=true);
    aicM(n) = icF.aic;
    bicM(n) = icF.bic;
    aiccM(n) = icF.aicc;
    caicM(n) = icF.caic;
    hqcM(n) = icF.hqc;
    rr = R2M(n);
    if mean(E{n} - E{n - 1}) < 0.001
        break
    end

    for j = 1:length(LogData)
        mdl = modelM{n};
        clear yy stdY
        for i = 1:length(mdl)
            if strcmp(class(mdl{i}),'RegressionGP')
                [yy(i,:),stdY(i,:)] = predict(mdl{i},LogData{j}(:,2:end)); %(predict(mdl,LogData{j}) + 1)/2*(MAX - MIN) + MIN;
            else
                yy(i,:) = predict(mdl{i},LogData{j}(:,2:end)); %(predict(mdl,LogData{j}) + 1)/2*(MAX - MIN) + MIN;
                stdY(i,1:size(yy,2)) = 0;
            end
        end
        yy = mean(yy,1);
        yLog{j} = yLog{j} + yy';
        yLogSTD{j} = mean(stdY',2);
    end
end

close all
ResIC = [aicM',bicM',aiccM',caicM',hqcM'];
% % BiasM = (BiasM - min(BiasM))/(max(BiasM) - min(BiasM));
% % VarianceM = (VarianceM - min(VarianceM))/(max(VarianceM) - min(VarianceM));
% % cd(directory)
figure(100)
plot(Y,yp,'o','MarkerEdgeColor',[0,0,0.5])
hold on
plot(Y,Y,'LineWidth',2,'Color','r')
xlim([min(Y),max(Y)])
ylim([min(Y),max(Y)])
grid on
xlabel('real values')
ylabel('estimated values')
set(gca,'fontname','times','fontsize', 14) ;
nn = ['Final Additive Models',num2str(name)];
print(fullfile(direction,nn), '-dpng', '-r1200');
%
figure(200)
yyaxis left
plot(1:n,bias,'-o','LineWidth',2,'Color',[0,0,0.5],'MarkerEdgeColor',[0,0,0.5],'MarkerFaceColor',[0,0,0.5])
hold on
plot(1:n,VarianceM,'-*','LineWidth',2)
grid on
yyaxis right
plot(1:n,bias.^2+VarianceM,'-d')

lgd = legend;
lgd.Color = 'none';
xlabel('iteration')
legend('Bias','Variance','MSE',Location='southwest');
set(gca,'fontname','times','fontsize', 14)
nn = ['Bias Variance TO Final Additive Models',num2str(name)];
print(fullfile(direction,nn), '-dpng', '-r1200');

figure(300)
plot(1:n,ResIC,'-o','MarkerEdgeColor','auto','MarkerFaceColor','auto')
grid on
legend('AIC','BIC','AICC','CAIC','HQC');
lgd = legend;
lgd.Color = 'none';
xlabel('iteration')
set(gca,'fontname','times','fontsize', 14) ;
nn = ['Information Criteria Final Additive Models',num2str(name)];
print(fullfile(direction,nn), '-dpng', '-r1200');

figure(400)
plot(1:n,R2M,'-o','LineWidth',2,'Color',[0,0,0.5],'MarkerEdgeColor',[0,0,0.5],'MarkerFaceColor',[0,0,0.5])
hold on
grid on
xlabel('iteration')
ylabel('Adj R^2')
set(gca,'fontname','times','fontsize', 14) ;
nn = ['adjusted R squered Final Additive Models',num2str(name)];
print(fullfile(direction,nn), '-dpng', '-r1200');

figure(500)
plot(1:n,mseM,'-o','LineWidth',2,'Color',[0,0,0.5],'MarkerEdgeColor',[0,0,0.5],'MarkerFaceColor',[0,0,0.5])
hold on
grid on
xlabel('iteration')
ylabel('MSE')
set(gca,'fontname','times','fontsize', 14) ;
nn = ['MSE Final Additive Models',num2str(name)];
print(fullfile(direction,nn), '-dpng', '-r1200');
end