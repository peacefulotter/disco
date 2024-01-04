import { dataset, training, Task, TaskProvider, TrainingSchemes } from '..'

export const simpleFace: TaskProvider = {
    getTask(): Task {
        return {
            taskID: 'simple_face',
            displayInformation: {
                taskTitle: 'Simple Face',
                summary: {
                    preview: 'Can you detect if the person in a picture is a child or an adult?',
                    overview: 'Simple face is a small subset of face_task from Kaggle',
                },
                limitations: 'The training data is limited to small images of size 200x200.',
                tradeoffs: 'Training success strongly depends on label distribution',
                dataFormatInformation: '',
                dataExampleText: 'Below you find an example',
                dataExampleImage:
                    'https://storage.googleapis.com/deai-313515.appspot.com/example_training_data/simple_face-example.png',
            },
            trainingInformation: {
                modelID: 'simple_face-model',
                epochs: 50,
                modelURL:
                    'https://storage.googleapis.com/deai-313515.appspot.com/models/mobileNetV2_35_alpha_2_classes/model.json',
                roundDuration: 1,
                validationSplit: 0.2,
                batchSize: 10,
                preprocessingFunctions: [dataset.ImagePreprocessing.Normalize],
                learningRate: 0.001,
                modelCompileData: {
                    optimizer: 'sgd',
                    loss: 'categoricalCrossentropy',
                    metrics: ['accuracy'],
                },
                dataType: 'image',
                IMAGE_H: 200,
                IMAGE_W: 200,
                LABEL_LIST: ['child', 'adult'],
                scheme: TrainingSchemes.FEDERATED, // secure aggregation not yet implemented for federated
                noiseScale: undefined,
                clippingRadius: undefined,
            },
        }
    },

    async getModel(): Promise<training.model.Model> {
        throw new Error('Not implemented')
    },
}
