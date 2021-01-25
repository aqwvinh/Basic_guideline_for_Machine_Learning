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
However, these lines of code above only save the model itself and not the X_train (which you'd spent time to create also).
That'd be useful to save them too in order to reuse the model ! It's possible by saving them in a tuple.
```
tuple_objects = (model, X_train, y_train, score)
# Save tuple
pickle.dump(tuple_objects, open("pricing_tuple_rf.pkl", 'wb'))
# Restore tuple
pickled_model, pickled_X_train, pickled_y_train, pickled_score = pickle.load(open("tuple_model.pkl", 'rb'))
```
