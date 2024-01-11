import { tf, training } from '@epfml/discojs-core'
import { AdamW, clipByGlobalNormObj } from './optimizers'
import { Wandb } from './wandb'
import { GPTConfig } from './model'

export async function train(
    model: any,
    ds: any,
    config: GPTConfig,
    callbacks: training.TrainingCallbacks
): Promise<void> {
    console.log(tf.getBackend())

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

    const wandb = new Wandb({
        ...config,
        platform:
            typeof window !== 'undefined' &&
            typeof window.document !== 'undefined'
                ? 'browser'
                : 'node',
        gpu: 'nvidia-4070-ti',
        model: config.modelType,
    })

    callbacks.onTrainBegin()

    let epoch = 1
    let iteration = 1
    let iterator = await ds.iterator()

    const start = Date.now()
    let time = start

    while (true) {
        // Get new batch of x and y
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

        // Calculates loss, computes gradients and applies them
        const loss = tf.tidy(() => {
            let { grads, value: loss } = opt.computeGradients(() => {
                const logits = model.apply(xs)
                const loss = tf.losses.softmaxCrossEntropy(ys, logits)
                return loss as tf.Scalar
            })
            let gradsClipped = clipByGlobalNormObj(grads, 1)
            opt.applyGradients(gradsClipped)
            return loss
        })

        const lossVal = await loss.array()

        callbacks.onBatchEnd(iteration)

        // Log wandb
        wandb.log({
            'train/perplexity': Math.exp(lossVal),
            'train/loss': lossVal,
            iter: iteration,
            mem: tf.memory().numBytes,
            dt_ms: Date.now() - time,
            time_s: (Date.now() - start) / 1000,
        })
        if (iteration % 100 === 0) {
            console.log(iteration, Date.now() - time)
        }
        time = Date.now()

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
    wandb.finish()
}
