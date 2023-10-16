import { z } from 'zod'

const envSchema = z.object({
  OPENAI_API_KEY: z.string(),
})

export function getEnv() {
  return envSchema.parse(process.env)
}
