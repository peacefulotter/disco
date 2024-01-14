class Deferred<T> {
    promise: Promise<T>
    resolve: (value: T | PromiseLike<T>) => void = () => {}
    reject: (reason?: any) => void = () => {}

    constructor() {
        this.promise = new Promise<T>((resolve, reject) => {
            this.resolve = resolve
            this.reject = reject
        })
    }
}

export class Cache<E> {
    position: number = 0
    private readonly cache: Deferred<E>[]

    private constructor(
        private readonly length: number,
        private readonly request: (pos: number) => void | Promise<void>
    ) {
        this.cache = Array.from({ length }, () => new Deferred<E>())
    }

    // pre-loads the cache with the first n requests
    static async init<E>(
        length: number,
        request: (pos: number) => void | Promise<void>,
        initializer: (c: Cache<E>) => void
    ): Promise<Cache<E>> {
        const cache = new Cache<E>(length, request)
        initializer(cache)
        for (let pos = 0; pos < length; pos++) {
            cache.request(pos)
        }
        return cache
    }

    put(pos: number, elt: E): void {
        const promise = this.cache[pos] as Deferred<E>
        promise.resolve(elt)
    }

    async next(): Promise<E> {
        const eltOrDeffered = this.cache[this.position]
        // const time = Date.now()
        //console.time(`${time} ${this.position}`)
        console.time('cache')
        const elt = await eltOrDeffered.promise
        // console.timeEnd(`${time} ${this.position}`)
        console.timeEnd('cache')
        const pos = this.position
        this.cache[pos] = new Deferred<E>()
        this.request(pos)
        this.position = (pos + 1) % this.length
        return elt
    }
}
