import { tf, dataset } from '../..'

export class NodeTabularLoader extends dataset.loader.TabularLoader<string> {
    async loadDatasetFrom(
        source: string,
        csvConfig: Record<string, unknown>
    ): Promise<dataset.Dataset> {
        const prefix = 'file://'
        if (source.slice(0, 7) !== prefix) {
            source = prefix + source
        }
        return tf.data.csv(source, csvConfig)
    }
}
