function agent = createSACAgent(obsInfo, actInfo)
% SAC Agent with stochastic actor (dual outputs) and twin RNN critics

% Actor network (LSTM-based)
inPath = [
    sequenceInputLayer(prod(obsInfo.Dimension), Name="obsIn")
    lstmLayer(128, OutputMode="sequence", Name="actorLSTM")
    ];

meanPath = [
    fullyConnectedLayer(prod(actInfo.Dimension), Name="meanFC")
    sigmoidLayer(Name="meanOutLyr")
    ];
sdevPath = [
    fullyConnectedLayer(prod(actInfo.Dimension), Name="stdFC")
    softplusLayer(Name="stdOutLyr")
    ];

actorNet = dlnetwork;
actorNet = addLayers(actorNet, inPath);
actorNet = addLayers(actorNet, meanPath);
actorNet = addLayers(actorNet, sdevPath);
actorNet = connectLayers(actorNet, "actorLSTM", "meanFC");
actorNet = connectLayers(actorNet, "actorLSTM", "stdFC");
actorNet = initialize(actorNet);

actor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ...
    ActionMeanOutputNames="meanOutLyr",...
    ActionStandardDeviationOutputNames="stdOutLyr",...
    ObservationInputNames="obsIn");

% Critic builder (RNN-based)
    function criticNet = buildCritic()
        obsPath = [
            sequenceInputLayer(prod(obsInfo.Dimension), Name="obsInCrit")
            lstmLayer(128, OutputMode="sequence", Name="obsLSTM")
            reluLayer(Name="obsRelu")
            ];
        actPath = [
            sequenceInputLayer(prod(actInfo.Dimension), Name="actInCrit")
            lstmLayer(128, OutputMode="sequence", Name="actLSTM")
            reluLayer(Name="actRelu")
            ];
        commonPath = [
            additionLayer(2, Name="add")
            reluLayer(Name="commonRelu")
            fullyConnectedLayer(128, Name="commonFC")
            reluLayer(Name="commonRelu2")
            fullyConnectedLayer(1, Name="qValue")
            ];
        lgraph = layerGraph();
        lgraph = addLayers(lgraph, obsPath);
        lgraph = addLayers(lgraph, actPath);
        lgraph = addLayers(lgraph, commonPath);
        lgraph = connectLayers(lgraph, 'obsRelu', 'add/in1');
        lgraph = connectLayers(lgraph, 'actRelu', 'add/in2');
        criticNet = dlnetwork(lgraph);
        criticNet = initialize(criticNet);
    end

criticNet1 = buildCritic();
criticNet2 = buildCritic();

critic1 = rlQValueFunction(criticNet1, obsInfo, actInfo, ...
    ObservationInputNames="obsInCrit", ActionInputNames="actInCrit");
critic2 = rlQValueFunction(criticNet2, obsInfo, actInfo, ...
    ObservationInputNames="obsInCrit", ActionInputNames="actInCrit");

agent = rlSACAgent(actor, [critic1 critic2]);

outputDir = '/Users/mohammadfarid/Desktop/results geochemistry/actor critic nets';  % Change this path% Create directory if it doesn't exist
agent_name = 'SAC';
fig_name = sprintf('%s-Actor Net', agent_name);
[G, h, fig] = plotNetworkAsGraph(actorNet, ...
    'SavePath', outputDir, ...
    'SaveName', fig_name, ...
    'SaveFormats', {'fig', 'jpeg', 'tif'}, ...
    'Resolution', 1200, ...
    'NodeFontSize', 14, ...
    'MarkerSize', 10, ...
    'NodeColor', [0.2, 0.6, 0.8], ...
    'LineWidth', 2, ...
    'Layout', 'layered', ...
    'title',fig_name,...
    'ShowLabels', true, ...
    'BackgroundColor', [0.95, 0.95, 0.95], ...
    'GridOn', true);
fig_name = sprintf('%s-Critic Net', agent_name);
[G, h, fig] = plotNetworkAsGraph(criticNet1, ...
    'SavePath', outputDir, ...
    'SaveName', fig_name, ...
    'SaveFormats', {'fig', 'jpeg', 'tif'}, ...
    'Resolution', 1200, ...
    'NodeFontSize', 14, ...
    'MarkerSize', 10, ...
    'NodeColor', [0.2, 0.6, 0.8], ...
    'LineWidth', 2, ...
    'Layout', 'layered', ...
    'title',fig_name,...
    'ShowLabels', true, ...
    'BackgroundColor', [0.95, 0.95, 0.95], ...
    'GridOn', true);

end
