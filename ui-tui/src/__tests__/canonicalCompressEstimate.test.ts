import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'

const WINDOW = 262_144

beforeEach(() => resetUiState())

describe('canonical /compress', () => {
  const info = { model: 'm', skills: {}, tools: {}, execution_epoch: '1', execution_generation: 4 }
  const flush = () => new Promise(resolve => setImmediate(resolve))

  const run = async (summary: Record<string, unknown>) => {
    patchUiState({
      info,
      sid: 'owner',
      usage: {
        calls: 2,
        compressions: 0,
        context_max: WINDOW,
        context_percent: 57,
        context_used: 150_000,
        input: 1,
        output: 1,
        total: 2
      }
    })
    let resumes = 0

    const request = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        resumes += 1
        const generation = resumes === 1 ? 4 : 5

        return {
          session_id: 'owner',
          revision: 8,
          execution_generation: generation,
          info: { ...info, execution_generation: generation },
          messages: [{ role: 'assistant', text: 'summary' }]
        }
      }

      return { session_id: 'owner', operation: 'compress', execution_generation: 5, summary }
    })

    const ctx = {
      slashFlightRef: { current: 0 },
      gateway: { gw: { isCanonical: true, request }, rpc: request },
      transcript: { sys: vi.fn(), setHistoryItems: vi.fn() },
      local: { maybeWarn: vi.fn() },
      session: { resumeById: vi.fn() },
      composer: { enqueue: vi.fn() }
    }

    createSlashHandler(ctx as any)('/compress')
    await flush()
    await flush()

    return getUiState().usage
  }

  it("shows the authority's compacted request estimate, marked, never 0", async () => {
    const usage = await run({ headline: 'Compressed: 364 → 119 messages', after_tokens: 65_583 })

    expect(usage).toMatchObject({
      compressions: 1,
      context_estimated: true,
      context_max: WINDOW,
      context_percent: 25,
      context_used: 65_583
    })
  })
})
