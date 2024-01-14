import fs from 'fs'
import path from 'path'
import esMain from 'es-main'
import { readdir } from 'fs/promises'
import { encode } from 'gpt-tokenizer/model/text-davinci-003'

// For ts-node-esm
import { fileURLToPath } from 'url'
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const BATCH_LENGTH = 4096
export const TOKENIZED_FILE_EXTENSION = 'tokens'

// TODO: support for multiple files being tokenized together into a single file

async function getFileStreams(datasetDir: string) {
    const files = await readdir(datasetDir)
    const preprocessFiles = files.filter(
        (file) =>
            !file.endsWith('zip') && !file.endsWith(TOKENIZED_FILE_EXTENSION)
    )
    console.log(
        'Found',
        preprocessFiles.length,
        'files to preprocess:',
        preprocessFiles
    )
    const streams = preprocessFiles.map((file) => ({
        file,
        getStream: async () =>
            new Promise<fs.ReadStream>((resolve) => {
                const stream = fs.createReadStream(
                    path.join(datasetDir, file),
                    {
                        encoding: 'utf8',
                        highWaterMark: 1,
                        fd: undefined,
                    }
                )
                stream.on('readable', () => resolve(stream))
            }),
    }))
    return streams
}

const preprocessStream = async (
    datasetDir: string,
    file: string,
    getStream: () => Promise<fs.ReadStream>
) => {
    const stream = await getStream()

    const writeFilePath = path.join(
        datasetDir,
        file + '.' + TOKENIZED_FILE_EXTENSION
    )
    console.log('Writing to', writeFilePath)
    const writeFileStream = fs.createWriteStream(writeFilePath)

    let accumulator: string[] = []
    let char: string

    while (null !== (char = stream.read(1))) {
        accumulator.push(char)
        if (accumulator.length >= BATCH_LENGTH && char === ' ') {
            const chunk = accumulator.join('')
            const tokens = encode(chunk)
            const array = new Uint16Array(tokens)
            const buffer = Buffer.from(array.buffer)
            writeFileStream.write(buffer)
            accumulator = []
        }
    }

    writeFileStream.end()
}

export default async function preprocess() {
    const datasetDir = path.join(__dirname, 'datasets', 'wikitext-103')
    console.log('Preprocessing step located at:', datasetDir)
    const streams = await getFileStreams(datasetDir)

    for await (const { file, getStream } of streams) {
        const label = `Preprocessing ${file}`
        console.time(label)
        await preprocessStream(datasetDir, file, getStream)
        console.timeEnd(label)
    }
}

if (esMain(import.meta)) {
    await preprocess()
}
