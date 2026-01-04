function runMultiAgentNAS()
clear, close all, clc
% Set up paths and data
% Load and preprocess data
data = readtable("TOC_Tmax_HI_OI.xlsx");
data = table2array(data);
MAX = max(data);
MIN = min(data);
data = (data - MIN)./(MAX - MIN);
idx = randperm(size(data,1),size(data,1));
X_dl = data(idx,1:5);
Y_dl = data(idx,6:end);
% Agent configurations
agent_configs = {
    {'PG', @createPGAgent},
    {'A2C', @createA2CAgent},
    {'A3C', @createA3CAgent},
    {'AC', @createACAgent},
    {'PPO', @createPPOAgent},
    {'TRPO', @createTRPOAgent},
    {'DDPG', @createDDPGAgent},
    {'TD3', @createTD3Agent},
    {'SAC', @createSACAgent},
    {'DDPGES', @createDDPGAgent},
    {'TD3ES', @createTD3Agent},
    {'SACES', @createSACAgent}
    };

% Training parameters
training_episodes = 10;  % Reduced for demonstration

% Results storage
all_results = struct();

% Train each agent
for i = 1:length(agent_configs)
    % myCluster = parcluster('Processes')
    % delete(myCluster.Jobs)
    % parpool; % Create new pool (adjust profile/size if needed)
    agent_name = agent_configs{i}{1};
    agent_creator = agent_configs{i}{2};
    % Create results directory
    base_path = ['G:\My Drive\results geochemistry','\',agent_name];
    if ~exist(base_path, 'dir')
        mkdir(base_path);
    end
    fprintf('Training %s agent...\n', agent_name);
    % try
    % Create environment for this agent
    env = FullyConnectedEnvMultiAgent(X_dl, Y_dl, 'glorot', MAX, MIN, 5, base_path, agent_name);
    obsInfo = getObservationInfo(env);
    actInfo = getActionInfo(env);
    % Create agent
    agent = agent_creator(obsInfo, actInfo);
    % Train agent
    if contains(agent_name, 'ES')
        % Use evolutionary strategy for DDPG, TD3, SAC
        results = trainWithEvolutionaryStrategy(agent, env, agent_name,base_path);
    else
        % Use standard training
        results = trainStandardAgent(agent, env, agent_name, training_episodes,base_path);
    end

    % Store results
    all_results.(agent_name) = results;

    % Save individual agent results
    save(fullfile(base_path, sprintf('%s_results.mat', agent_name)), 'results');
    %
    % catch ME
    %     fprintf('Error training %s: %s\n', agent_name, ME.message);
    %     continue;
    % end
end
base_path = '/Users/mohammadfarid/Desktop/results geochemistry';
if ~exist(base_path, 'dir')
    mkdir(base_path);
end
% Save combined results
save(fullfile(base_path, 'all_agent_results.mat'), 'all_results');

% Generate comparison
generateComparison(all_results, X_dl, Y_dl, MAX, MIN, base_path);
end
