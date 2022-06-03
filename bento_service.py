from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput
from bentoml.frameworks.transformers import TransformersModelArtifact

@env(pip_packages=["transformers==4.19.2", "torch==1.11.0+cu113"])
@artifacts([TransformersModelArtifact("gptj_model")])
class TransformerService(BentoService):
    @api(input=JsonInput(), batch=False)
    def predict(self, parsed_json):
        src_text = parsed_json.get("text")

        model = self.artifacts.gptj_model.get("model")
        tokenizer = self.artifacts.gptj_model.get("tokenizer")

        input_ids = tokenizer(src_text, return_tensors="pt").input_ids

        gen_ids = model.generate(input_ids,
            do_sample=True,
            max_length=100,
            temperature=0.9,
            # use_cache=True
        )

        output = tokenizer.batch_decode(gen_ids)[0]

        return output