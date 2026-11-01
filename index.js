import fs from 'fs';
import { pipeline } from "@huggingface/transformers";
// import { normalize } from 'path';
import 'dotenv/config';
// config();
// console.log(process.env.GROQ_API_KEY);


// const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
// step 1 chunking 

const text = fs.readFileSync('./docs/sample.txt','utf-8');

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

const chunks = chunkText(text);
const vectorStore = [];
for (const chunk of chunks) {
  vectorStore.push({ text: chunk, vector: await embed(chunk) });
}
console.log(`Indexed ${vectorStore.length} chunks.\n`);
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

async function retrieve(questions, topK = 3) {
    const qVec = await embed(questions);
    return vectorStore
        .map((item) => ({ text: item.text, score: similarity(qVec, item.vector) }))
        .sort((a, b) => b.score - a.score)
        .filter((e) => e.score >= 0.05)
        .slice(0, topK);
}

// const results = await retrieve("How many leave days do I get?");
// console.log(results);

function buildPrompt(question, contextChunks) {
    const context = contextChunks.map((c, i) => `[${i + 1}] ${c.text}`).join("\n\n");
    return `Answer the question with long response using ONLY the context below.
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
await ask("Hello there, what does this company do?");