import { expect } from 'chai'
import * as api from '../src/helpers/communication/federated/api'

const task = 'titanic'
const user = 'a'
const nonValidTask = 'nonValidTask'
const oldVersion = -1

describe('API test', () => { // the tests container
  it('Connect to valid task', async () => {
    expect((await api.connect(task, user)).status).equal(200)
  })

  it('disconnect from valid task', async () => {
    expect((await api.disconnect(task, user)).status).equal(200)
  })

  it('Connect to non valid task', async () => {
    api.connect(nonValidTask, user).catch(function (error) {
      expect(error.response.status).equal(404)
    })
  })

  it('versionIsOld true for old version', async () => {
    await api.connect(task, user)
    const resp = await api.getIsVersionOld(task, user, oldVersion)
    expect(resp.data.versionIsOld).true
    await api.disconnect(task, user)
  })
})
