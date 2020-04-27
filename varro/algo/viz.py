import numpy as np
import plotly.graph_objects as go

def plot_preds(x_true,
                y_true,
                y_pred):
  '''
  TODO: animation

  '''

  fig = go.Figure()

  fig.add_trace(go.Scatter(x=x_true,
                         y=y_true),
                         name='true',
                         mode='lines'))
  fig.add_trace(go.Scatter(x=x_true,
                          y=y_pred,
                          name='pred',
                          mode='markers'))

  fig.show()
