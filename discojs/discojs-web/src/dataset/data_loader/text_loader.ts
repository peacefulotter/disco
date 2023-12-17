import { tf, data } from '../..'
import {
    TextConfig,
    TextLoader,
    TokenizedDataset,
} from '@epfml/discojs-core/src/dataset/data_loader/text_loader'

export class WebTextLoader extends TextLoader<File> {
    load(source: File, config: TextConfig): Promise<TokenizedDataset> {
        throw new Error('Method not implemented.')
    }
    loadAll(source: File, config?: Partial<TextConfig> | undefined): Promise<data.DataSplit> {
        throw new Error('Method not implemented.')
    }
    async loadDatasetFrom(source: File, config?: TextConfig): Promise<data.Dataset> {
        const file = new tf.data.FileDataSource(source)
        return new tf.data.TextLineDataset(file)
    }
}
