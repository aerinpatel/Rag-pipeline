import fs from 'fs';
import { pipeline } from "@huggingface/transformers";
import pg from 'pg';
import crypto from 'crypto';
import path from 'path';
import 'dotenv/config';
// config();
// console.log(process.env.GROQ_API_KEY);

// console.log(process.env.DATABASE_URL);
// console.log(pg.Query("SELECT COUNT(*) FROM chunks"));
const pool = new pg.Pool({connectionString:process.env.DATABASE_URL});
// const r = await pool.query("SELECT COUNT(*) FROM chunks");
// console.log(r.rows);
// console.log(process.env.DATABASE_URL?.replace(/:[^:@]*@/, ":****@"));
// const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
// step 1 chunking 

// const text = fs.readFileSync('./docs/sample.txt','utf-8');

function chunkText(text, size = 200, overlap = 50){
    const chunks = [];
    for(let i = 0;i < text.length;i += size-overlap){
        const piece = text.slice(i,i+size).trim();
        if(piece) chunks.push(piece);
    }
    return chunks;
}

// ---------- 3. EMBED ----------
// An embedding model turns text into a vector (array of 384 numbers).
// Similar meaning => similar numbers. Runs locally, no API key needed.

const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
async function embed(str) {
    const out = await embedder(str,{pooling:"mean",normalize:true})
    return Array.from(out.data);
}

// const chunks = chunkText(text);
// const vectorStore = [];
// for (const chunk of chunks) {
//   vectorStore.push({ text: chunk, vector: await embed(chunk) });
// }
// console.log(`Indexed ${vectorStore.length} chunks.\n`);
// console.log(vectorStore);
 
function similarity(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
//   console.log(sum);
  return sum;
}
 
// const q = await embed("How many holidays do I get?");

// similarity(q, await embed("Employees receive 24 paid leave days"));          // yours: 0.345
// similarity(q, await embed("A lost laptop must be reported within 24 hours")); 
// similarity(q, await embed("Internet bill reimbursement is capped at 1000 rupees"));
// similarity(q, await embed("The probation period is 3 months"));

async function retrieve(question, topK = 3) {
    const qVec = await embed(question);
    const vecStr = `[${qVec.join(",")}]`;

    const { rows } = await pool.query(
        `SELECT c.content, d.source, 1 - (c.embedding <=> $1) AS score
         FROM chunks c
         JOIN documents d ON d.id = c.document_id
         ORDER BY c.embedding <=> $1
         LIMIT $2`,
        [vecStr, topK]
    );

    return rows
        .map(r => ({ text: r.content, source: r.source, score: Number(r.score) }))
        .filter(r => r.score >= 0.05);
}

// const results = await retrieve("How many leave days do I get?");
// console.log(results);

function buildPrompt(question, contextChunks) {
    const context = contextChunks.map((c, i) => `[${i + 1}] ${c.text}`).join("\n\n");
    return `Answer the question with response using ONLY the context below.
    If the answer is not in the context, say "I don't know".
    
    Context:
    ${context}
    
    Question: ${question}`;
}

async function askLLM(prompt){
    try{
        const res = await fetch('https://api.groq.com/openai/v1/chat/completions',
            {
                method:'POST',
                headers: { "Content-Type": "application/json",'Authorization': `Bearer ${process.env.GROQ_API_KEY}` },
                
                body: JSON.stringify({
                    model: "openai/gpt-oss-20b",
                    // stream: false,
                    messages: [{ role: "user", content: prompt }],
                }),
            }
        )
        if (!res.ok) {
            const err = await res.text();
            return `API error ${res.status}: ${err}`;
        }
        const data = await res.json();
        return data.choices[0].message.content;
    }catch(e){
        return "api is facting some kind of error so check it out for this prompt: " + prompt;
    }
}
// console.log(await askLLM("Say hello in one full sentence"));

async function ask(question){
    const context = await retrieve(question);
    console.log(context);
    if(context.length === 0){
        console.log("I don't know");
        return;
    }
    const prompt = buildPrompt(question,context);

    console.log("LLM: ", await askLLM(prompt));
}
// await ask("Hello there, what does this company do?");


async function indexFile(filePath) {
    const source = path.basename(filePath);
    const fileText = fs.readFileSync(filePath, 'utf-8');
    const hash = crypto.createHash('sha256').update(fileText).digest('hex');

    // Is this file already indexed, and is it unchanged?
    const existing = await pool.query(
        "SELECT id, content_hash FROM documents WHERE source = $1", [source]
    );
    if (existing.rows.length && existing.rows[0].content_hash === hash) {
        console.log(`${source}: unchanged, skipping.`);
        return;
    }

    // Do the slow work (embedding) BEFORE touching the database
    const chunks = chunkText(fileText);
    const vectors = [];
    for (const chunk of chunks) vectors.push(await embed(chunk));

    // Write everything in one transaction: all or nothing
    const client = await pool.connect();
    try {
        await client.query("BEGIN");

        if (existing.rows.length) {
            // File was edited: remove the old version (chunks go with it)
            await client.query("DELETE FROM documents WHERE id = $1", [existing.rows[0].id]);
        }

        const doc = await client.query(
            "INSERT INTO documents (source, content_hash) VALUES ($1, $2) RETURNING id",
            [source, hash]
        );
        const documentId = doc.rows[0].id;

        for (let i = 0; i < chunks.length; i++) {
            await client.query(
                "INSERT INTO chunks (document_id, chunk_index, content, embedding) VALUES ($1, $2, $3, $4)",
                [documentId, i, chunks[i], `[${vectors[i].join(",")}]`]
            );
        }

        await client.query("COMMIT");
        console.log(`${source}: indexed ${chunks.length} chunks.`);
    } catch (e) {
        await client.query("ROLLBACK");
        throw e;
    } finally {
        client.release();
    }
}

async function indexAll(folder = './docs') {
    for (const file of fs.readdirSync(folder)) {
        if (file.endsWith('.txt')) await indexFile(path.join(folder, file));
    }
}

await indexAll();
await ask("how many leave days do i get?");
await pool.end();