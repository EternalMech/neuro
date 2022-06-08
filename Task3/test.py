import plotly.express as px
import pandas as pd
import openpyxl

pd.pandas.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)
pd.options.display.expand_frame_repr = False
# pd.set_option("precision", 2)

final_df = pd.read_excel('outputs/Task3UltraPro_georges_predict.xlsx')  # загружаем результаты анализа
print(final_df)


df_plot = final_df[['activation', 'dropout', 'batch_size', 'epochs', 'input_shape', 'accuracy']]

fig = px.parallel_coordinates(df_plot,
                              color="accuracy",
                              range_color=[df_plot['accuracy'].min(), df_plot['accuracy'].max()],
                              title='Зависимость точности от гиперпараметров нейронной сети',
                              color_continuous_scale=[
                                  (0.00, "gray"),   (0.75, "gray"),
                                  (0.75, "orange"), (1.00, "orange")
                              ])

fig.write_html("outputs/Task3UltraPro_georges_predict.html")  # сохраняем в файл
fig.show()