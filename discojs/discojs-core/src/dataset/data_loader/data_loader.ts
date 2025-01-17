import { Task } from '../../'
import { Dataset } from '../dataset'
import { Data, DataSplit } from '../data'

export interface DataConfig {
    features?: string[]
    labels?: string[]
    shuffle?: boolean
    validationSplit?: number
    inference?: boolean
}

export abstract class DataLoader<Source, Config = DataConfig> {
    constructor(protected task: Task) {}

    abstract createData(dataset: Dataset, size?: number): Promise<Data>

    abstract load(source: Source, config: Config): Promise<Dataset>

    abstract loadAll(sources: Source | Source[], config: Config): Promise<DataSplit>
}
