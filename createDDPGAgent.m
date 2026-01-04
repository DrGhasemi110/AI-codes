function agent = createDDPGAgent(obsInfo, actInfo)
% DDPG Agent with deterministic actor

% Actor network (RNN-based)
actorNet = [
    sequenceInputLayer(prod(obsInfo.Dimension), Name="obsIn")
    lstmLayer(128, Name="actorLSTM")
    fullyConnectedLayer(prod(actInfo.Dimension), Name="actorFC")
    sigmoidLayer(Name="actorSigmoid")
    ];
actorNet = dlnetwork(layerGraph(actorNet));
actorNet = initialize(actorNet);
actor = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo);

% Critic network with matching RNN structure for obs and action
% Observation path
obsPath = [
    sequenceInputLayer(prod(obsInfo.Dimension), Name="obsInCrit")
    lstmLayer(128, Name="obsLSTM")
    fullyConnectedLayer(128, Name="obsFC")
    reluLayer(Name="obsRelu")
    ];

% Action path
actPath = [
    sequenceInputLayer(prod(actInfo.Dimension), Name="actInCrit")
    lstmLayer(128, Name="actLSTM")
    fullyConnectedLayer(128, Name="actFC")
    reluLayer(Name="actRelu")
    ];

% Combined path
commonPath = [
    additionLayer(2, Name="add")
    reluLayer(Name="commonRelu")
    fullyConnectedLayer(128, Name="commonFC")
    reluLayer(Name="commonRelu2")
    fullyConnectedLayer(1, Name="qValue")  % Single output
    ];

% Build layer graph
lgraph = layerGraph();
lgraph = addLayers(lgraph, obsPath);
lgraph = addLayers(lgraph, actPath);
lgraph = addLayers(lgraph, commonPath);

% Connect paths
lgraph = connectLayers(lgraph, 'obsRelu', 'add/in1');
lgraph = connectLayers(lgraph, 'actRelu', 'add/in2');

criticNet = dlnetwork(lgraph);
criticNet = initialize(criticNet);
critic = rlQValueFunction(criticNet, obsInfo, actInfo, ...
    ObservationInputNames='obsInCrit', ActionInputNames='actInCrit');

% Create DDPG agent
agent = rlDDPGAgent(actor, critic);
% agent.AgentOptions.NoiseOptions.Variance = 0.1;
% agent.AgentOptions.NoiseOptions.VarianceDecayRate = 1e-5;

outputDir = '/Users/mohammadfarid/Desktop/results geochemistry/actor critic nets';  % Change this path% Create directory if it doesn't exist
if ~isfolder(outputDir)
    mkdir(outputDir);
end
agent_name = 'DDPG';
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
[G, h, fig] = plotNetworkAsGraph(criticNet, ...
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
