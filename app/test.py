import utility.util as ut
import plotly.express as px


# # load data
# disaster_df = ut.read_db('../data/DisasterResponse.db', 'Disaster')
#
# genre_counts = disaster_df.groupby('genre').count()['message']
# genre_names = list(genre_counts.index)
#
# fig = px.bar(
#     x=genre_names,
#     y=genre_counts
# )
#
# ut.whats(fig)
#
# print(str(fig))