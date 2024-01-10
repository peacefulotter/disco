import { tf, training } from '@epfml/discojs-core'
import { AdamW, clipByGlobalNormObj } from './optimizers'

interface TrainConfig {
    epochs?: number | null
    maxIter?: number | null
    batchSize: number
    shuffle?: boolean | 'batch' | number
    lr?: number
    weightDecay?: boolean | number
    verbose?: boolean
}

export async function train(
    model: any,
    ds: any,
    config: TrainConfig,
    callbacks: training.TrainingCallbacks
): Promise<void> {
    const defaultConfig: TrainConfig = {
        epochs: null,
        maxIter: null,
        batchSize: 16,
        shuffle: true,
        lr: 6e-4,
        weightDecay: false,
    }
    config = Object.assign(defaultConfig, config || {})

    // if (config.shuffle === true) {
    //     ds = ds.shuffle(config.batchSize * 10)
    // } else if (config.shuffle === 'batch') {
    //     ds = ds.shuffle(config.batchSize)
    // } else if (config.shuffle && !isNaN(config.shuffle)) {
    //     ds = ds.shuffle(config.shuffle)
    // }
    // ds = ds.batch(config.batchSize)

    var includeInWeightDecay: string[] = []
    var excludeFromWeightDecay: string[] = []

    if (config.weightDecay === true) {
        config.weightDecay = 1e-4
    }
    let opt: tf.Optimizer
    if (config.weightDecay) {
        model.getNamedWeights().forEach((v: any) => {
            if (
                v.name.includes('bias') ||
                v.name.includes('normalization') ||
                v.name.includes('emb')
            ) {
                excludeFromWeightDecay.push(v.name)
            } else {
                includeInWeightDecay.push(v.name)
            }
        })
        opt = new AdamW({
            learningRate: config.lr,
            weightDecayRate: config.weightDecay,
            includeInWeightDecay,
            excludeFromWeightDecay,
        })
    } else {
        opt = tf.train.adam(config.lr)
    }

    callbacks.onTrainBegin()

    let epoch = 1
    let iteration = 1
    let iterator = await ds.iterator()

    while (true) {
        callbacks.onBatchBegin(iteration)
        let next = await iterator.next()
        if (next.done) {
            callbacks.onEpochEnd(epoch)
            epoch++
            if (config.epochs && epoch > config.epochs) {
                break
            }
            callbacks.onEpochBegin(epoch)
            iterator = await ds.iterator()
            next = await iterator.next()
        }
        const { xs, ys } = next.value
        console.log('gpt training loop, next:', xs.shape, ys.shape)

        // Keep loss for reporting
        const optFunc = () => {
            const logits = model.apply(xs)
            const loss = tf.keep(tf.losses.softmaxCrossEntropy(ys, logits))
            return loss
        }
        const loss = tf.tidy(() => {
            const loss = optFunc()
            let { grads } = opt.computeGradients(() => loss as any) as any
            let gradsClipped = clipByGlobalNormObj(grads, 1)
            opt.applyGradients(gradsClipped)
            return loss
        })

        let lossVal = await loss.array()
        console.warn('gpt training loop, loss:', lossVal)

        callbacks.onBatchEnd(iteration)

        // Dispose everything
        loss.dispose()
        xs.dispose()
        ys.dispose()

        // Check if we should stop
        iteration++
        if (config.maxIter && iteration > config.maxIter) {
            break
        }

        if (config.verbose) {
            console.log('Mem:', tf.memory())
            console.log(`Epoch: ${epoch}, Step: ${iteration}, Loss: ${lossVal}`)
        }

        await new Promise((resolve) => setTimeout(resolve, 1))
    }

    callbacks.onTrainEnd()
}
