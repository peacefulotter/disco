import { tf, data } from '../..'
import { TextConfig, TextLoader } from '@epfml/discojs-core/src/dataset/data_loader/text_loader'

export class WebTextLoader extends TextLoader<File, tf.data.TextLineDataset> {
    async load(source: File, config: TextConfig): Promise<tf.data.TextLineDataset> {
        const src = source
        const file = new tf.data.FileDataSource(src)
        return new tf.data.TextLineDataset(file)
    }
    async loadAll(
        source: File[],
        config?: Partial<TextConfig> | undefined
    ): Promise<data.DataSplit> {
        // TODO: multiple source or just one?
        // TODO: if one source, just return the train dataset?
        // TODO: return a TextLineDataset and not a TokenizedDataset??
        console.log('WebTextLoader.loadAll', source.length, config)
        const datasets = await Promise.all(
            source.map(async (s) => await this.load(s, config as TextConfig))
        )
        const dataset = datasets[0]
        return {
            train: await data.TextData.init(dataset, this.task),
            validation: await data.TextData.init(dataset, this.task),
        }
    }
}
