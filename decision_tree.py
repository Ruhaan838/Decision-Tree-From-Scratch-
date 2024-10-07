import pandas as pd
from typing import List, Tuple, AnyStr, Dict
import math
from tqdm import tqdm
from graphviz import Digraph

class Node:
    def __init__(self, name: str, gain: float, attributes: List, parent: object = None):
        
        self.name = name
        self.gain = gain
        self.attributes = attributes
        self.parent = parent
        self.children = {}

    def add_child(self, value, node):
        self.children[value] = node

class DataProcessing:
    def train_label(self, data: pd.DataFrame, train_idx: List, label_idx: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        train_data = data.loc[:, train_idx]
        label_data = data.loc[:, label_idx]
        return train_data, label_data

class DecisionTreeID3:
    def __init__(self, train_data: pd.DataFrame, label_data: pd.DataFrame) -> None:
        
        self.train_data = train_data
        self.label_data = label_data
        self.train_columns_list = list(train_data.columns)
        
        if len(label_data.columns) > 1:
            raise ValueError("Label should have only 1 column!")
        else:
            self.label_columns_list = list(label_data.columns)
        
        self.train_columns_set_values = {col: list(set(train_data[col])) for col in self.train_columns_list}
        self.label_columns_set_values = {col: list(set(label_data[col])) for col in self.label_columns_list}

    def __calculate_p_n(self, train_data: pd.DataFrame, label_data: pd.DataFrame, train_column: AnyStr, train_entity: AnyStr, label_column: AnyStr, label_entity: AnyStr) -> Tuple[int, int]:
        
        filtered_train = train_data[train_data[train_column] == train_entity]
        filtered_label = label_data.loc[filtered_train.index]
        
        p = sum(filtered_label[label_column] == label_entity)
        n = len(filtered_label) - p
        
        return p, n

    def __entropy_pn(self, p: int, n: int) -> float:
        
        if p + n == 0:
            return 0
        
        total = p + n
        term1 = p / total if p != 0 else 0
        term2 = n / total if n != 0 else 0
        
        entropy = 0
        
        if term1 > 0:
            entropy -= term1 * math.log(term1, 2)
        if term2 > 0:
            entropy -= term2 * math.log(term2, 2)
            
        return entropy

    def __information_entropy(self, subset_size: int, total_size: int, entropy: float) -> float:
        
        return (subset_size / total_size) * entropy

    def __gain(self, total_entropy: float, weighted_entropy_sum: float) -> float:
        
        return total_entropy - weighted_entropy_sum

    def __make_decision(self, row_attribute_gain: Dict, parent: Node = None) -> Node:
        
        max_gain_attr = max(row_attribute_gain, key=row_attribute_gain.get)
        max_gain = row_attribute_gain[max_gain_attr]
        max_gain_values = self.train_columns_set_values[max_gain_attr]
        
        return Node(max_gain_attr, max_gain, max_gain_values, parent)

    def __calculate_gains(self, train_data: pd.DataFrame, label_data: pd.DataFrame, total_pn: int, total_entropy: float, label_column: AnyStr, label_entity: AnyStr) -> Dict:
        
        row_attribute_gain = {}
        for col in train_data.columns:
            weighted_entropy_sum = 0
            for val in tqdm(self.train_columns_set_values[col], bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', desc=f"For {col}"):
                p, n = self.__calculate_p_n(train_data, label_data, col, val, label_column, label_entity)
                entropy = self.__entropy_pn(p, n)
                pn_total = p + n
                weighted_entropy_sum += self.__information_entropy(pn_total, total_pn, entropy)
            row_attribute_gain[col] = self.__gain(total_entropy, weighted_entropy_sum)
            
        return row_attribute_gain

    def __update_dataset(self, train_data: pd.DataFrame, label_data: pd.DataFrame, attribute: str, value: AnyStr) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        filtered_train = train_data[train_data[attribute] == value].drop(columns=[attribute])
        filtered_label = label_data.loc[filtered_train.index]
        
        return filtered_train, filtered_label

    def __print_tree(self, node: Node, level: int = 0) -> None:
        
        indent = "  " * level 
        
        if node:
            print(f"{indent}{node.name} (Gain: {node.gain:.4f})")
            
            for attribute_value, child_node in node.children.items():
                
                print(f"{indent}  |── {attribute_value}")
                self.__print_tree(child_node, level + 1)

    def __visualize_tree(self, root: Node) -> None:
        
        dot = Digraph(comment="Decision Tree")
        
        def add_nodes_edges(node, parent_name=None, edge_label=None):
            node_name = f'{node.name}\n(Gain: {node.gain:.4f})'
            dot.node(node_name)

            if parent_name is not None:
                dot.edge(parent_name, node_name, label=str(edge_label)) 
                
            for value, child in node.children.items():
                if child:
                    add_nodes_edges(child, node_name, edge_label=value)  
            
        add_nodes_edges(root)
        return dot

    def fit(self, verbose=0, visualize_tree=0):
        
        label_col = self.label_columns_list[0]
        label_entity = self.label_columns_set_values[label_col][1]  
        
        p = sum(self.label_data[label_col] == label_entity)
        n = len(self.label_data) - p
        
        total_pn = p + n
        total_entropy = self.__entropy_pn(p, n)
        
        def build_tree(train_data: pd.DataFrame, label_data: pd.DataFrame, parent_node: Node = None):
            
            if len(set(label_data[label_col])) == 1:
                return
            
            row_attribute_gain = self.__calculate_gains(train_data, label_data, total_pn, total_entropy, label_col, label_entity)
            node = self.__make_decision(row_attribute_gain, parent_node)
            
            for val in node.attributes:
                filtered_train, filtered_label = self.__update_dataset(train_data, label_data, node.name, val)
                if len(filtered_train) > 0:  
                    child_node = build_tree(filtered_train, filtered_label, node)
                    node.add_child(val, child_node)
            
            return node

        root = build_tree(self.train_data, self.label_data)
        print()
        print("Decision Tree built.")
        
        if verbose == 1:
            print()
            self.__print_tree(root)
        
        if visualize_tree == 1:
            tree_graph = self.__visualize_tree(root)
            tree_graph.render("decision_tree", format="png")
            tree_graph.view()

if __name__ == "__main__":
    data = {
        
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rainy", "Rainy", "Rainy", "Overcast", "Sunny", "Sunny", "Rainy", "Sunny", "Overcast", "Overcast", "Rainy"],
    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "High", "High"],
    "Windy": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
    "PlayTennis": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
    }

    df = pd.DataFrame(data)

    train_columns = ["Outlook", "Temperature", "Humidity", "Windy"]
    label_column = ["PlayTennis"]

    data_processor = DataProcessing()
    train_data, label_data = data_processor.train_label(df, train_columns, label_column)

    tree = DecisionTreeID3(train_data, label_data)
    tree.fit(verbose=1, visualize_tree=0)
