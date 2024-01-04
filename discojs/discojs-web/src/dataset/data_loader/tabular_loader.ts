import { tf, dataset } from '../..'

export class WebTabularLoader extends dataset.loader.TabularLoader<File> {
    async loadDatasetFrom(
        source: File,
        csvConfig: Record<string, unknown>
    ): Promise<dataset.Dataset> {
        return new tf.data.CSVDataset(new tf.data.FileDataSource(source), csvConfig)
    }
}
