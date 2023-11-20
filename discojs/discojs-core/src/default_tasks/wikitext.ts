import { tf, data, Task, TaskProvider } from "../index.node";
// FIXME: can't resolve ".."

import { model } from "gpt-tfjs";
const { GPTLMHeadModel } = model;

export const wikitext: TaskProvider = {
  getTask(): Task {
    return {
      taskID: "wikitext-103",
      displayInformation: {
        taskTitle: "wikitext-103-raw",
        summary: {
          preview:
            "In this challenge, we ask you to do next word prediction on a dataset of Wikipedia articles.",
          overview:
            "Wikitext-103-raw is a dataset comprising unprocessed text excerpts from Wikipedia articles, designed for tasks related to natural language processing and language modeling.",
        },
        limitations:
          "The dataset may contain noise, inconsistencies, and unstructured content due to its raw nature, potentially posing challenges for certain NLP tasks.",
        tradeoffs:
          "The raw format may lack structured annotations and may require additional preprocessing for specific applications.",
        dataFormatInformation:
          "The dataset is organized as a large text file, with each line representing a segment of raw text from Wikipedia articles.",
        dataExampleText:
          'An example excerpt from the dataset could be: "The history of artificial intelligence dates back to ancient times, with philosophical discussions on the nature of thought and reasoning."',
      },
      trainingInformation: {
        modelID: "wikitext-103-raw-model",
        epochs: 10,
        roundDuration: 10,
        validationSplit: 0.2,
        batchSize: 10,
        modelCompileData: {
          optimizer: "sgd",
          loss: "categoricalCrossentropy",
          metrics: ["perplexity"],
        },
        dataType: "text",
        preprocessingFunctions: [data.TextPreprocessing.Tokenize, data.TextPreprocessing.Padding],
        scheme: "Decentralized",
        noiseScale: undefined,
        decentralizedSecure: true,
        minimumReadyPeers: 3,
        maxShareValue: 100,
      },
    };
  },

  async getModel(): Promise<tf.LayersModel> {
    const gpt = GPTLMHeadModel({});
    return gpt.model;
  },
};
