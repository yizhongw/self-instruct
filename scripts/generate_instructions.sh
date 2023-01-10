batch_dir=data/gpt3_generations/

python self_instruct/bootstrap_instructions.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate 50000 \
    --seed_tasks_path data/seed_tasks.jsonl \
    --engine "davinci"