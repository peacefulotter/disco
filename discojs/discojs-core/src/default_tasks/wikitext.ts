import { tf, training, Task, TaskProvider, TrainingSchemes } from '..'
import { model as gpt } from '@epfml/gpt-tfjs'
import { TFJSModel } from '../training/model'

const modelConfig: gpt.GPTConfig = {
    modelType: 'gpt-nano',
    epochs: 10,
    maxIter: 10_000,
    batchSize: 4,
    blockSize: 128,
    lr: 0.001,
    vocabSize: 50257,
    evaluate: true,
    maxEvalBatches: 12,
    evaluateEvery: 100,
} as const

export const wikitext: TaskProvider<gpt.GPTConfig> = {
    getTask(): Task<gpt.GPTConfig> {
        return {
            id: 'wikitext-103',
            displayInformation: {
                taskTitle: 'wikitext-103-raw',
                summary: {
                    preview:
                        'In this challenge, we ask you to do next word prediction on a dataset of Wikipedia articles.',
                    overview:
                        'Wikitext-103-raw is a dataset comprising unprocessed text excerpts from Wikipedia articles, designed for tasks related to natural language processing and language modeling.',
                },
                limitations:
                    'The dataset may contain noise, inconsistencies, and unstructured content due to its raw nature, potentially posing challenges for certain NLP tasks.',
                tradeoffs:
                    'The raw format may lack structured annotations and may require additional preprocessing for specific applications.',
                dataFormatInformation:
                    'The dataset is organized as a large text file, with each line representing a segment of raw text from Wikipedia articles.',
                dataExampleText:
                    'An example excerpt from the dataset could be: "The history of artificial intelligence dates back to ancient times, with philosophical discussions on the nature of thought and reasoning."',
            },
            trainingInformation: {
                dataType: 'text',
                modelID: 'wikitext-103-raw-model',
                validationSplit: 0.2, // FIXME: is this used somewhere? because train, eval and test are already split in dataset
                maxIterations: modelConfig.maxIter,
                epochs: modelConfig.epochs ?? 1,
                batchSize: modelConfig.batchSize,
                learningRate: modelConfig.lr,
                modelCompileData: {
                    optimizer: 'sgd',
                    loss: 'categoricalCrossentropy',
                    metrics: ['precision', 'mse'], // 'perplexity' doesnt exist
                },
                modelConfig,
                /**
                 * preprocessing is done prior to training so it is not needed in my case
                 * but otherwise, one can use the following template to use a custom tokenizer
                 * and the predefined preprocessing functions
                 */
                // import tokenizer from 'gpt-tokenizer/model/text-davinci-003'
                // ...
                // tokenizer,
                // preprocessingFunctions: [
                //     data.TextPreprocessing.Tokenize,
                //     data.TextPreprocessing.Padding,
                // ],
                // vocabSize: 50257
                // blockSize: 64
                scheme: TrainingSchemes.FEDERATED,
                noiseScale: undefined,
                decentralizedSecure: true,
                minimumReadyPeers: 3,
                maxShareValue: 100,
                roundDuration: 10,
            },
        }
    },

    async getModel(): Promise<training.model.Model> {
        console.log('GPT Config:', modelConfig)
        // const model = new gpt.GPTLMHeadModel(config)
        const model = gpt.GPT(modelConfig)
        return new TFJSModel(this.getTask(), model as any as tf.LayersModel)
    },
}
