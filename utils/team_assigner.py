from sklearn.cluster import KMeans
import numpy as np
import cv2
from sklearn.decomposition import PCA


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self,image):                                
        # Reshape the image to 2D array and find out 2 main clusters in it (Background and player)
        image_2d = image.reshape(-1,3)

        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=15)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):                                          # Then we get the player color scheme, by substracting colors in the "background" segment
                                                                                    # The background cluster is determined by seeing which cluster occurs more frequently in the corners
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        image = cv2.GaussianBlur(image, (5, 5), 0)

        top_half_image = image[0:int(image.shape[0]/2),:]

        # # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self,frame, player_detections):                           # creates 2 clusters for all player colors, and then finds the mean color for future assignments. 

        
        player_colors = [self.get_player_color(frame, player["bbox"]) for player in player_detections.values()]
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=15).fit(player_colors)
        self.team_colors = {i + 1: center for i, center in enumerate(kmeans.cluster_centers_)}
        self.kmeans = kmeans


    def get_player_team(self,frame,player_bbox,player_id):                          # add "team_id" to the player if not yet added and return it, otherwise just return it
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)                     # warning pops out, because referee is wearing all black which is different from all other objects


        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1


        self.player_team_dict[player_id] = team_id

        return team_id
