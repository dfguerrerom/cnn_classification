from box import Box

settings = Box({
    "test_1" : {
        "dataset" : {
            "variable" : "degraded_forest" 
        },
        "batch_size": 16, 
        "rescale_factor": 224,
        "model": {
            "name" : "dpn92",
            "fixed_feature": False,
            "out_features": 2,
        },
        "optimizer": {
            "name": "SDG",
            "lr" : 0.003*32/128,  # 32 batch size
            "momentum":0.9
        },
        "loss_fn" : {
            "name": "CrossEntropy",
        },
        "scheduler" : {
            "name" : "StepLR",
            "step_size": 2, 
            "gamma": 0.2,
        }
    },
    "test_2" : {
        "dataset" : {
            "variable" : "degraded_forest" 
        },
        "batch_size": 8, 
        "rescale_factor": 224,
        "model": {
            "name" : "dpn131",
            "fixed_feature": False,
            "out_features": 2,
        },
        "optimizer": {
            "name": "SDG",
            "lr" : 0.003*32/128,  # 32 batch size
            "momentum":0.9
        },
        "loss_fn" : {
            "name": "CrossEntropy",
        },
        "scheduler" : {
            "name" : "StepLR",
            "step_size": 2, 
            "gamma": 0.2,
        }
    },
    "test_3" : {
        "dataset" : {
            "variable" : "lc_tags" 
        },
        "batch_size": 8, 
        "rescale_factor": 224,
        "model": {
            "name" : "dpn92",
            "fixed_feature": False,
            "out_features": 2,
        },
        "optimizer": {
            "name": "SDG",
            "lr" : 0.003*32/128,  # 32 batch size
            "momentum":0.9
        },
        "loss_fn" : {
            "name": "CrossEntropy",
        },
        "scheduler" : {
            "name" : "StepLR",
            "step_size": 2, 
            "gamma": 0.2,
        }
    },
    "test_4" : {
        "description":"We are using an rgb combination [2,1,0]",
        "dataset" : {
            "variable" : "lc_tags" 
        },
        "batch_size": 128, 
        "rescale_factor": 32,
        "model": {
            "name" : "resnet34",
            "transfer": True,
            "fixed_feature": False, # If not fixed feature is fine tune but if transfer true
            "out_features": 6,
        },
        "optimizer": {
            "name": "Adam",
            "lr" : 0.01,
            "weight_decay":0.0001
        },
        "loss_fn" : {
            "name": "CrossEntropy",
        },
        "scheduler" : {
            "name" : "LambdaLR",
            "lr_lambda" : lambda epoch: 0.65 ** epoch
        }
    },
    "mnist" : {
        "dataset" : {
            "variable" : "lc_tags" 
        },
        "batch_size": 8, 
        "rescale_factor": 32,
        "model": {
            "name" : "resnet34",
            "fixed_feature": True,
            "out_features": 10,
        },
        "optimizer": {
            "name": "Adam",
            "lr" : 0.1,
            "weight_decay":0.0001
        },
        "loss_fn" : {
            "name": "CrossEntropy",
        },
        "scheduler" : {
            "name" : "LambdaLR",
            "lr_lambda" : lambda epoch: 0.65 ** epoch
        }
    },
})