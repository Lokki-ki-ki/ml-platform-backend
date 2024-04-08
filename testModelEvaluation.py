from app.modelEvaluator import MlModel

example = {
    "clientsToSubmissions":{"1":"QmdfzatZMtMmaWMTNsJuvCQcBbHAQFSGTrYLi6CiZ5fWTi","2":"QmX4T5dLBrubvGmUJnkM1tP1j8oPC5ZwDMUWdDR1R9wWNN","3":"QmTCxkXPG9fetQs1mP6QmQDWAx5vUc325u8AndPquJQv62","4":"QmPZeG6uJsX1EsKd4BeR5cPNBScAwpU7CLXRGXrEgbUpSV","5":"Qmc8VCb4DWZbXxhqcCCsqdVoYiGqrmkVbQ4fHPhYopRwS1"},
    "clientsToReputation":{"1":"100","2":"100","3":"100", "4": "100", "5": "100"},
    "rewardPool":"999999999999999900",
    "modelAddress":"QmYEMkTVdYF7bBoJ28D2Lrqex1xozLZ5yHQ8pjDuJ18zQe",
    "testDataAddress":"Qmeo6yw83vLYX3zkgPDiGK24FB99pz1PX1EngLHLPyko76",
    "testLabelAddress":"QmWTBKyjTsuHHq6kHdCnbvF3PQZDQa2pincoyAc8DzisFX",
    "testDataHash":"64d59ad605ca4af2b947844984c54f11409928c2ad9880f864ee7459bb17e308",
    "testLabelHash":"ae416418d6466f81e7a63f0d8c09400fa221c6389eca6c67d7b12370715c385c"
}
test = MlModel(example["clientsToSubmissions"], example["testDataAddress"], example["testLabelAddress"], example["modelAddress"], example["rewardPool"], example["clientsToReputation"], example["testDataHash"], example["testLabelHash"])

    