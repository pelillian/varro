import numpy as np
import plotly.graph_objects as go

def plot_preds(y_pred,
               x_true=np.linspace(-2*np.pi, 2*np.pi, 100),
               y=np.sin):
  '''
  TODO: animation

  - Args
      y_pred (np arr):
          Model outputs of x_true
      x_true (np arr):
          True domain
      y (fn):
          Maps x to range
  '''

  fig = go.Figure()

  fig.add_trace(go.Scatter(x=x_true,
                         y=y(x_true),
                         name='true',
                         mode='lines'))
  fig.add_trace(go.Scatter(x=x_true,
                          y=y_pred,
                          name='pred',
                          mode='markers'))

  fig.show()
