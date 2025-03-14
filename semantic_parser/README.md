This is an experiment to observe the performance of LLMs as semantic parsers. 

1. Get the data
   
    `wget https://www.cs.cmu.edu/~pengchey/reg_attn_data.zip`
   
    `unzip reg_attn_data.zip`
 
The folder `data` will be available under /home/<username>


2. Create pyenv with all required files


    1. install python libraries using `python3 -m pip install -r requirements.txt`
    2. download ollama to run local models like MistralAI 7B
   
        a. `curl -fsSL https://ollama.com/install.sh | sh`
       
        b. download mistral models
            `ollama run mistral`
            `ollama run mistral-large`


3. I used `data\smcalflow_cs\calflow.orgchart.event_create\source_domain_with_target_num128\test.jsonl` as test file for running the small model and `sample_for_large_model.jsonl` for running the large model


4. to run the small model `python model.py s`


5. to run the large model `python model.py l`


6. 'model_output.jsonl` has the output predictions from both the small and large model for comparison
