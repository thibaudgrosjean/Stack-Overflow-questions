from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from predict import Predictor

app = FastAPI()
predictor = Predictor()


class TextIn(BaseModel):

    title: str
    body: str


@app.post('/predict')
def predict(payload: TextIn):

    data_title = payload.title
    data_body = payload.body
    prediction = predictor.predict(data_title, data_body)
    response = JSONResponse(
        status_code=200,
        content={'suggested_tags': prediction})

    return response


# predictor.predict('c++ having a problem','i want to vectorize ios program')