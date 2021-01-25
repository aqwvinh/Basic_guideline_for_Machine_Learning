# Deploy the model
Once you've trained your model and found the best results, you need to save (pickle) the model to be able to reuse it
```
# save the model to disk
import pickle
filename = 'task_model.sav'
pickle.dump(model, open(filename, 'wb'))
```
Then later, load the model
```
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_valid, y_valid)
print(f"Test result: {round(result*100,2)} %")
```
