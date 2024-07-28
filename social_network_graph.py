import heapq
import networkx as nx  # Used for visualizing the social network graph
import matplotlib.pyplot as plt  # Used for plotting the network graph

# Class representing a user in the social network
class User:
    def __init__(self, user_id, name, interests=None, posts=None):
        self.user_id = user_id
        self.name = name
        self.interests = interests if interests else []
        self.posts = posts if posts else []
        self.friends = set()  # Initialize an empty set for friends

    def update_profile(self, name=None, interests=None, posts=None):
        if name:
            self.name = name
        if interests:
            self.interests = interests
        if posts:
            self.posts = posts

    def __repr__(self):
        return f"User({self.user_id}, {self.name}, Interests: {self.interests}, Posts: {self.posts})"

# Class representing the social network graph
class SocialNetworkGraph:
    def __init__(self):
        self.users = {}  # Dictionary to store users by user_id
        self.graph = {}  # Adjacency list to store friendships

    def add_user(self, user_id, name, interests=None, posts=None):
        if user_id in self.users:
            print(f"User ID {user_id} already exists.")
            return
        user = User(user_id, name, interests, posts)
        self.users[user_id] = user
        self.graph[user_id] = set()

    def remove_user(self, user_id):
        if user_id not in self.users:
            print(f"User ID {user_id} does not exist.")
            return
        # Remove user from friends' lists
        for friend_id in self.graph[user_id]:
            self.graph[friend_id].remove(user_id)
        del self.graph[user_id]
        del self.users[user_id]

    def add_relationship(self, user_id1, user_id2):
        if user_id1 not in self.users or user_id2 not in self.users:
            print("Both users must exist in the network.")
            return
        self.graph[user_id1].add(user_id2)
        self.graph[user_id2].add(user_id1)

    def remove_relationship(self, user_id1, user_id2):
        if user_id1 not in self.users or user_id2 not in self.users:
            print("Both users must exist in the network.")
            return
        if user_id2 in self.graph[user_id1]:
            self.graph[user_id1].remove(user_id2)
        if user_id1 in self.graph[user_id2]:
            self.graph[user_id2].remove(user_id1)

    def update_user_profile(self, user_id, name=None, interests=None, posts=None):
        if user_id not in self.users:
            print(f"User ID {user_id} does not exist.")
            return
        self.users[user_id].update_profile(name, interests, posts)

    # Breadth-First Search (BFS) to traverse the social network graph
    def bfs(self, start_user_id):
        visited = set()
        queue = [start_user_id]
        while queue:
            user_id = queue.pop(0)
            if user_id not in visited:
                print(f"Visited {user_id}")
                visited.add(user_id)
                queue.extend(self.graph[user_id] - visited)

    # Depth-First Search (DFS) to traverse the social network graph
    def dfs(self, start_user_id):
        visited = set()
        stack = [start_user_id]
        while stack:
            user_id = stack.pop()
            if user_id not in visited:
                print(f"Visited {user_id}")
                visited.add(user_id)
                stack.extend(self.graph[user_id] - visited)

    # Dijkstra's algorithm to find the shortest path from the start user to all other users
    def dijkstra(self, start_user_id):
        distances = {user_id: float('inf') for user_id in self.graph}
        distances[start_user_id] = 0
        priority_queue = [(0, start_user_id)]
        while priority_queue:
            current_distance, current_user_id = heapq.heappop(priority_queue)
            if current_distance > distances[current_user_id]:
                continue
            for neighbor in self.graph[current_user_id]:
                distance = current_distance + 1
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))
        return distances

    # Finding connected components in the network
    def connected_components(self):
        visited = set()
        components = []
        for user_id in self.graph:
            if user_id not in visited:
                component = self._dfs_component(user_id, visited)
                components.append(component)
        return components

    # Helper method for DFS to find connected components
    def _dfs_component(self, start_user_id, visited):
        stack = [start_user_id]
        component = []
        while stack:
            user_id = stack.pop()
            if user_id not in visited:
                visited.add(user_id)
                component.append(user_id)
                stack.extend(self.graph[user_id] - visited)
        return component

    # Merge sort for sorting users based on a key function
    def merge_sort_users(self, key_func):
        user_list = list(self.users.values())
        if len(user_list) <= 1:
            return user_list
        mid = len(user_list) // 2
        left_half = self.merge_sort_users_helper(user_list[:mid], key_func)
        right_half = self.merge_sort_users_helper(user_list[mid:], key_func)
        return self.merge(left_half, right_half, key_func)

    # Helper method for merge sort
    def merge_sort_users_helper(self, user_list, key_func):
        if len(user_list) <= 1:
            return user_list
        mid = len(user_list) // 2
        left_half = self.merge_sort_users_helper(user_list[:mid], key_func)
        right_half = self.merge_sort_users_helper(user_list[mid:], key_func)
        return self.merge(left_half, right_half, key_func)

    # Merging two sorted lists for merge sort
    def merge(self, left, right, key_func):
        sorted_list = []
        while left and right:
            if key_func(left[0]) <= key_func(right[0]):
                sorted_list.append(left.pop(0))
            else:
                sorted_list.append(right.pop(0))
        sorted_list.extend(left if left else right)
        return sorted_list

    # Quick sort for sorting users based on a key function
    def quick_sort_users(self, key_func):
        user_list = list(self.users.values())
        if len(user_list) <= 1:
            return user_list
        pivot = user_list[0]
        less_than_pivot = [user for user in user_list[1:] if key_func(user) <= key_func(pivot)]
        greater_than_pivot = [user for user in user_list[1:] if key_func(user) > key_func(pivot)]
        return self.quick_sort_users_helper(less_than_pivot, key_func) + [pivot] + self.quick_sort_users_helper(greater_than_pivot, key_func)

    # Helper method for quick sort
    def quick_sort_users_helper(self, user_list, key_func):
        if len(user_list) <= 1:
            return user_list
        pivot = user_list[0]
        less_than_pivot = [user for user in user_list[1:] if key_func(user) <= key_func(pivot)]
        greater_than_pivot = [user for user in user_list[1:] if key_func(user) > key_func(pivot)]
        return self.quick_sort_users_helper(less_than_pivot, key_func) + [pivot] + self.quick_sort_users_helper(greater_than_pivot, key_func)

    # Binary search for finding a user based on a key function and value
    def binary_search_users(self, key_func, key_value):
        user_list = sorted(self.users.values(), key=key_func)
        left, right = 0, len(user_list) - 1
        while left <= right:
            mid = (left + right) // 2
            if key_func(user_list[mid]) == key_value:
                return user_list[mid]
            elif key_func(user_list[mid]) < key_value:
                left = mid + 1
            else:
                right = mid - 1
        return None

    # Calculate various statistics about the network
    def network_statistics(self):
        num_users = len(self.users)
        num_relationships = sum(len(friends) for friends in self.graph.values()) // 2
        avg_friends = num_relationships / num_users if num_users else 0
        density = 2 * num_relationships / (num_users * (num_users - 1)) if num_users > 1 else 0

        print(f"Number of users: {num_users}")
        print(f"Number of relationships: {num_relationships}")
        print(f"Average number of friends per user: {avg_friends:.2f}")
        print(f"Network density: {density:.4f}")

    # Calculate the clustering coefficient for a user
    def clustering_coefficient(self, user_id):
        friends = self.graph[user_id]
        if len(friends) < 2:
            return 0.0
        links = 0
        for friend1 in friends:
            for friend2 in friends:
                if friend1 != friend2 and friend2 in self.graph[friend1]:
                    links += 1
        possible_links = len(friends) * (len(friends) - 1)
        return links / possible_links

    # Recommend friends to a user based on mutual friends
    def recommend_friends(self, user_id):
        if user_id not in self.users:
            print(f"User ID {user_id} does not exist.")
            return []
        user_friends = self.graph[user_id]
        recommendations = {}
        for friend in user_friends:
            for mutual_friend in self.graph[friend]:
                if mutual_friend != user_id and mutual_friend not in user_friends:
                    if mutual_friend not in recommendations:
                        recommendations[mutual_friend] = 0
                    recommendations[mutual_friend] += 1
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [user_id for user_id, _ in sorted_recommendations]

    # Visualize the social network graph using networkx and matplotlib
    def visualize_graph(self):
        G = nx.Graph()
        for user_id, user in self.users.items():
            G.add_node(user_id, label=user.name)
        for user_id, friends in self.graph.items():
            for friend_id in friends:
                G.add_edge(user_id, friend_id)
        pos = nx.spring_layout(G)
        labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, pos, with_labels=True, labels=labels, node_color='skyblue', edge_color='gray', node_size=2000, font_size=10, font_color='black')
        plt.show()

    def __repr__(self):
        return f"SocialNetworkGraph(users={self.users}, graph={self.graph})"


# Example usage
if __name__ == "__main__":
    network = SocialNetworkGraph()
    network.add_user(1, "Ahmad", ["Reading", "Hiking"], ["Post 1"])
    network.add_user(2, "Georgio", ["Gaming", "Cooking"], ["Post 2"])
    network.add_user(3, "Charbel", ["Traveling", "Photography"], ["Post 3"])
    network.add_user(4, "Yara", ["Music", "Art"], ["Post 4"])
    network.add_user(5, "Hassan", ["Dancing", "Coding"], ["Post 5"])
    network.add_relationship(1, 2)
    network.add_relationship(1, 3)
    network.add_relationship(2, 4)
    network.add_relationship(3, 5)
    print(network)
    print("\nBFS starting from Ahmad:")
    network.bfs(1)
    print("\nDFS starting from Ahmad:")
    network.dfs(1)
    print("\nDijkstra's shortest paths from Ahmad:")
    distances = network.dijkstra(1)
    for user_id, distance in distances.items():
        print(f"Distance to {user_id}: {distance}")
    print("\nConnected components in the network:")
    components = network.connected_components()
    for component in components:
        print(component)
    print("\nUsers sorted by name (merge sort):")
    sorted_users_merge = network.merge_sort_users(key_func=lambda user: user.name)
    for user in sorted_users_merge:
        print(user)
    print("\nUsers sorted by name (quick sort):")
    sorted_users_quick = network.quick_sort_users(key_func=lambda user: user.name)
    for user in sorted_users_quick:
        print(user)
    print("\nSearch for user by ID (binary search):")
    user = network.binary_search_users(key_func=lambda user: user.user_id, key_value=3)
    print(user)
    print("\nNetwork statistics:")
    network.network_statistics()
    user_id = 1
    print(f"\nClustering coefficient for user {user_id}:")
    clustering_coef = network.clustering_coefficient(user_id)
    print(f"Clustering coefficient for user {user_id}: {clustering_coef:.4f}")
    user_id = 1
    print(f"\nFriend recommendations for user {user_id}:")
    recommendations = network.recommend_friends(user_id)
    print(recommendations)
    print("\nVisualizing the network graph:")
    network.visualize_graph()
