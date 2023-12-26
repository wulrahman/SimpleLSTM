% Assume you have a corpus as a cell array of sentences
% corpus = {'this is a sample sentence', 'another example sentence', 'yet another sentence'};

% Sample interaction as arrays
userInputs = {'how are you doing'};
chatbotResponses = {'I''m doing well how'};

corpus = [userInputs, chatbotResponses];

% Tokenize the corpus
tokens = cellfun(@strsplit, corpus, 'UniformOutput', false);

% Flatten tokens into a single list
allTokens = cat(2, tokens{:});

% Create a vocabulary of unique words
vocab = unique(allTokens);

% Define parameters
embeddingSize = 20;
learningRate = 0.01;
epochs = 1000;

% Define LSTM parameters
inputSize = 20;
hiddenSize = 20;
outputSize = 20;
sequenceLength = 4; % Length of input sequence
outputsequenceLength = 4; % Length of input sequence

% Initialize word embeddings randomly
wordEmbeddings = randn(embeddingSize, numel(vocab));

% Training loop (Skip-gram with negative sampling)
for epoch = 1:epochs
    for sentenceIndex = 1:numel(tokens)
        sentence = tokens{sentenceIndex};

        for targetIndex = 1:numel(sentence)
            targetWord = sentence{targetIndex};

            % Sample negative context words (in a real scenario, you'd sample these differently)
            negativeContextIndices = randi([1, numel(vocab)], 1, 5);

            % Positive context word
            contextWord = targetWord;

            % Update embeddings using gradient descent (skip-gram with negative sampling)
            targetEmbedding = wordEmbeddings(:, find(strcmp(vocab, targetWord)));
            contextEmbedding = wordEmbeddings(:, find(strcmp(vocab, contextWord)));
            negativeContextEmbeddings = wordEmbeddings(:, negativeContextIndices);

            % Calculate gradients (replace with actual gradient computation)
            gradientTarget = zeros(embeddingSize, 1);
            gradientContext = zeros(embeddingSize, 1);
            gradientNegativeContext = zeros(embeddingSize, numel(negativeContextIndices));

            % Update embeddings (replace with your actual update logic)
            wordEmbeddings(:, find(strcmp(vocab, targetWord))) = wordEmbeddings(:, find(strcmp(vocab, targetWord))) - learningRate * gradientTarget;
            wordEmbeddings(:, find(strcmp(vocab, contextWord))) = wordEmbeddings(:, find(strcmp(vocab, contextWord))) - learningRate * gradientContext;
            wordEmbeddings(:, negativeContextIndices) = wordEmbeddings(:, negativeContextIndices) - learningRate * gradientNegativeContext;
        end
    end
end

% Convert chatbot responses to embeddings
chatbotResponseEmbeddings = zeros(outputSize, outputsequenceLength);
for i = 1:numel(chatbotResponses)
    chatbotResponseEmbeddings = stringToEmbedding(chatbotResponses{i}, vocab, wordEmbeddings, outputSize);
end

% Convert chatbot responses to embeddings
userInputsEmbeddings = zeros(inputSize, sequenceLength);
for i = 1:numel(userInputs)
    userInputsEmbeddings = stringToEmbedding(userInputs{i}, vocab, wordEmbeddings, inputSize);
end

learningRate = 0.01;
numEpochs = 1000;

% Initialize LSTM parameters
Wf = randn(hiddenSize, inputSize);
Wi = randn(hiddenSize, inputSize);
Wc = randn(hiddenSize, inputSize);
Wo = randn(hiddenSize, inputSize);
bf = zeros(hiddenSize, 1);
bi = zeros(hiddenSize, 1);
bc = zeros(hiddenSize, 1);
bo = zeros(hiddenSize, 1);

Wout = randn(outputSize, hiddenSize);
bout = zeros(outputSize, 1);

% Generate sample data (replace this with your own data)
data = userInputsEmbeddings;
target = chatbotResponseEmbeddings;
% Use chatbot response embeddings as target

% Initialize arrays to store predictions and targets
allPredictions = zeros(outputSize, outputsequenceLength, numEpochs);
allTargets = zeros(outputSize, outputsequenceLength, numEpochs);

    
% Initialize a cell array to store losses
losses = cell(numEpochs, 1);

% Training loop
for epoch = 1:numEpochs
    % Forward pass
    h = zeros(hiddenSize, sequenceLength);
    c = zeros(hiddenSize, sequenceLength);
    y = zeros(outputSize, outputsequenceLength);

    for t = 1:sequenceLength
        ft = sigmoid(Wf * data(:, t) + bf);
        it = sigmoid(Wi * data(:, t) + bi);
        ct = tanh(Wc * data(:, t) + bc);
        ot = sigmoid(Wo * data(:, t) + bo);

        c(:, t+1) = ft .* c(:, t) + it .* ct;
        h(:, t+1) = ot .* tanh(c(:, t+1));
        y(:, t) = Wout * h(:, t) + bout;
    end
    
    % Store predictions and targets
    allPredictions(:, :, epoch) = y;
    allTargets(:, :, epoch) = target;

    % Compute loss
    loss = 0.5 * sum((y - target).^2);
    losses{epoch} = loss;

    % Backward pass (gradient descent)
    dWout = (y - target) * h(:, 1:end-1)';
    dbout = sum(y - target, 2);

    dWf = zeros(size(Wf));
    dWi = zeros(size(Wi));
    dWc = zeros(size(Wc));
    dWo = zeros(size(Wo));
    dbf = zeros(size(bf));
    dbi = zeros(size(bi));
    dbc = zeros(size(bc));
    dbo = zeros(size(bo));
    delta_next = zeros(hiddenSize, 1);

    for t = sequenceLength:-1:1
        delta = (Wout' * (y(:, t) - target(:, t))) + delta_next;

        df = delta .* tanh(c(:, t+1)) .* ft .* (1 - ft);
        di = delta .* ct .* it .* (1 - it);
        dc = delta .* it .* (1 - ct.^2);
        do = delta .* tanh(c(:, t+1)) .* ot .* (1 - ot);

        dWf = dWf + df * data(:, t)';
        dWi = dWi + di * data(:, t)';
        dWc = dWc + dc * data(:, t)';
        dWo = dWo + do * data(:, t)';
        dbf = dbf + df;
        dbi = dbi + di;
        dbc = dbc + dc;
        dbo = dbo + do;

        delta_next = Wf' * df + Wi' * di + Wc' * dc + Wo' * do;
    end

    % Update parameters
    Wf = Wf - learningRate * dWf;
    Wi = Wi - learningRate * dWi;
    Wc = Wc - learningRate * dWc;
    Wo = Wo - learningRate * dWo;
    bf = bf - learningRate * dbf;
    bi = bi - learningRate * dbi;
    bc = bc - learningRate * dbc;
    bo = bo - learningRate * dbo;
    Wout = Wout - learningRate * dWout;
    bout = bout - learningRate * dbout;

    % Display loss
    disp(['Epoch: ' num2str(epoch) ', Loss: ' num2str(loss)]);
end

% Convert cell array to a numeric array for plotting
lossesNumeric = cell2mat(losses);

% Plot the loss
figure;
subplot(2, 1, 1);
plot(1:numEpochs, lossesNumeric, '-o');
xlabel('Epoch');
ylabel('Loss');
title('Training Loss');

% Plot predictions and targets
subplot(2, 1, 2);
plot(1:sequenceLength, allTargets(1, :, end), '-o', 'DisplayName', 'Target');
hold on;
plot(1:sequenceLength, allPredictions(1, :, end), '-o', 'DisplayName', 'Prediction');
xlabel('Time Step');
ylabel('Value');
title('Target vs Prediction');
legend('show');

% Adjust layout
sgtitle('Training Loss and Predictions');

% inputMessage = preprocessInput("Hello, how are you? hell");
% disp(inputMessage);

% Tokenize and preprocess documents
tokens = cellfun(@strsplit, corpus, 'UniformOutput', false);
tokenizedDocuments = cat(2, tokens{:});

% Convert tokens to embeddings
documentEmbeddings = zeros(size(wordEmbeddings, 1), numel(tokenizedDocuments));
for i = 1:numel(tokenizedDocuments)
    wordIndices = find(ismember(vocab, tokenizedDocuments{i}));
    if ~isempty(wordIndices)
        documentEmbeddings(:, i) = mean(wordEmbeddings(:, wordIndices), 2);
    end
end

disp(documentEmbeddings);

function y = sigmoid(x)
    y = 1./(1 + exp(-x));
end


function wordEmbeddingsArray = stringToEmbedding(inputString, vocab, wordEmbeddings, truncationLength)
    % Tokenize the input string
    tokens = strsplit(lower(inputString), ' ');

    % Initialize an array to store embeddings for each word
    wordEmbeddingsArray = zeros(size(wordEmbeddings, 1), numel(tokens));

    % Iterate through tokens and convert to embeddings
    for i = 1:numel(tokens)
        word = tokens{i};
        if ismember(word, vocab)
            wordIndices = find(ismember(vocab, word));
            wordEmbeddingsArray(:, i) = wordEmbeddings(:, wordIndices);
        end
    end
   
end

