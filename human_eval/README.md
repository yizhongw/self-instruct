# Evaluating instruction following on more user-oriented data

To better access the practical value of instruction-following models, we curated a set of tasks and their instructions motivated by user-oriented applications (rather than well-studied NLP tasks). We aim to diversify the styles and formats of these tasks (e.g., instructions may be long or short; input/output may take the form of bullet points, tables, codes, equations, etc.). In total, we create 252 instructions with 1 instance per instruction. This data is used in the human evaluation section of [the self-instruct paper](https://arxiv.org/abs/2212.10560).

We open-source this evluation set here for the ease of evaluation and inspiration of future work on creating more comprehensive evaluation. **Please don't hack this evaluation set and don't use it for any training purpose!**

The files in the [`./predictions`](./predictions/) folder contain models' responses to these 252 instructions. Since these tasks indeed require a broad range of expertise and many of them are open-ended, we are still working on finding better ways for do automatic evaluation or systematic human evaluation. We will release the interface once it's finalized. Other ideas on better evaluating models' instruction-following capabilities are also super welcome!
