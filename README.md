# command

exec script on local computer

```
 python 02_train_as_job/train.py
```

exec job

```
az ml data create -f 02_train_as_job/data-train.yml
az ml data create -f 02_train_as_job/data-test.yml
az ml job create -f 02_train_as_job/job-train.yml
```

# API test

```
{
  "input_data": {
    "columns": [
      "vendorID",
      "lpepPickupDatetime",
      "passengerCount",
      "tripDistance",
      "pickupLongitude",
      "pickupLatitude",
      "dropoffLongitude",
      "dropoffLatitude"
    ],
    "index": [0,1],
    "data": [
      [2,1427148635.0,1,4.89,-73.99286651611328,40.68875503540039,-73.9989242553711,40.7445068359375],
      [2,1426510288.0,1,5.55,-73.98301696777344,40.693668365478516,-73.98097229003906,40.75892639160156]
      ]
  }
}
```