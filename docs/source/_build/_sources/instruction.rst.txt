How to use MELT?
=================

Running Pipeline
----------------

Run on local computer

.. code-block:: bash

    melt --mode generation \
               --model_name ura-hcmut/MixSUra \
               --dataset_name zalo_e2eqa \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42

Run on TGI

.. code-block:: bash

    melt --mode generation \
               --model_name ura-hcmut/MixSUra \
               --dataset_name zalo_e2eqa \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --tgi http://127.0.0.1:10025

Run on GPT (gpt-3.5-turbo, gpt-4)

.. code-block:: bash

    melt --mode generation \
               --model_name gpt-4 \
               --dataset_name zalo_e2eqa \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42

Evaluation
----------

.. code-block:: bash

    melt --mode evaluation \
               --model_name gpt-4 \
               --dataset_name zalo_e2eqa \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --output_dir results \
               --out_eval_dir out_new

End2End Pipeline
----------------

.. code-block:: bash

    melt --mode end2end \
               --model_name gpt-4 \
               --dataset_name zalo_e2eqa \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --output_dir results \
               --out_eval_dir out_new
