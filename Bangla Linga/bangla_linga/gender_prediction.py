import pickle
from pathlib import Path

script_location = Path(__file__).absolute().parent
data_loc = script_location / "name_gen_model"

from bangla_linga.BN_countvectorizer import CountVectorizer 
import bangla_linga.BN_ngram as ng

class BN_gen_pred(object):

  def __init__(self,model_name=data_loc):
    self.model_name = model_name
    with open(model_name, 'rb') as p:
      self.ob = pickle.load(p)


  def get_name_ara(self, name=None):
    gram_2 = ng.n_gram(name, 2)

    g2 = ' '.join(gram_2)
    gram_3 = ng.n_gram(name, 3)
    g3 = ' '.join(gram_3)

    name = [name + " " + g2 + " " + g3]

    ct = CountVectorizer()
    test = ct.transform(name)
    return test



  def predict_gender(self, name="None"):
    pred_gen = self.ob.predict(self.get_name_ara(name))

    if pred_gen == 0:
      return 'male'
    else: return 'female'