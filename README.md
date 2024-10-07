# Decision Tree from Scratch 

- This version of Decision Tree is completly build using algorithm [ID3 (Iterative Dichotomiser 3)](https://en.wikipedia.org/wiki/ID3_algorithm).

- In code Mostly work is done using `Pandas` and `math` 
    -  `Pandas` for Data manipulation.
    -  `math` for mathematical operations.
    
- It's also Construct the Tree using `Node` class.

- One simple visilization of tree is also there. (if verbose = 1)

         Outlook (Gain: 0.2467)
        |── Rainy
        Windy (Gain: 0.9403)
            |── Strong
            |── Weak
        |── Overcast
        |── Sunny
        Humidity (Gain: 0.9403)
            |── High
            |── Normal 
        

- Example data: 
    - Train Data : ` Outlook, Temperature, Humidity, Windy `
    - Label : ` PlayTennis `


    | Outlook   | Temperature | Humidity | Windy  | PlayTennis |
    |:-----------:|:-------------:|:----------:|:--------:|:------------:|
    | Sunny     | Hot         | High     | Weak   | No         |
    | Sunny     | Hot         | High     | Strong | No         |
    | Overcast  | Hot         | High     | Weak   | Yes        |
    | Rainy     | Mild        | High     | Weak   | Yes        |
    | Rainy     | Cool        | Normal   | Weak   | Yes        |
    | Rainy     | Cool        | Normal   | Strong | No         |
    | Overcast  | Cool        | Normal   | Strong | Yes        |
    | Sunny     | Mild        | High     | Weak   | No         |
    | Sunny     | Cool        | Normal   | Weak   | Yes        |
    | Rainy     | Mild        | Normal   | Weak   | Yes        |
    | Sunny     | Mild        | Normal   | Strong | Yes        |
    | Overcast  | Mild        | High     | Strong | Yes        |
    | Overcast  | Hot         | High     | Weak   | Yes        |
    | Rainy     | Mild        | High     | Strong | No         |

