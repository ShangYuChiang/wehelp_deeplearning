import math

# Task 1

# Base Geometric Object class for common functionalities
class GeometricObject:
    def __repr__(self):
        return f"{self.__class__.__name__}()"

# Define Point class
class Point(GeometricObject):
    def __init__(self, x, y):
        self.__x = x
        self.__y = y
    
    @property
    def x(self):
        return self.__x
    
    @x.setter
    def x(self, value):
        self.__x = value
    
    @property
    def y(self):
        return self.__y
    
    @y.setter
    def y(self, value):
        self.__y = value
    

    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"

# Define Line class
class Line(GeometricObject):
    def __init__(self, point1, point2):
        self.__point1 = point1
        self.__point2 = point2
    
    @property
    def point1(self):
        return self.__point1
    
    @property
    def point2(self):
        return self.__point2
    
    def slope(self):
        if self.point2.x - self.point1.x == 0:
            return None  # Vertical line
        return (self.point2.y - self.point1.y) / (self.point2.x - self.point1.x)
    
    def is_parallel(self, other_line):
        return self.slope() == other_line.slope()

    def is_perpendicular(self, other_line):
        slope1 = self.slope()
        slope2 = other_line.slope()
        if slope1 is None or slope2 is None:
            return slope1 == slope2  # One is vertical, check for perpendicularity
        return slope1 * slope2 == -1

# Define Circle class
class Circle(GeometricObject):
    def __init__(self, center, radius):
        self.__center = center
        self.__radius = radius
    
    @property
    def center(self):
        return self.__center
    
    @property
    def radius(self):
        return self.__radius
    
    def area(self):
        return math.pi * self.radius ** 2
    
    def intersects(self, other_circle):
        distance = math.sqrt((self.center.x - other_circle.center.x) ** 2 + (self.center.y - other_circle.center.y) ** 2)
        return distance < (self.radius + other_circle.radius)

# Define Polygon class
class Polygon(GeometricObject):
    def __init__(self, points):
        self.__points = points
    
    @property
    def points(self):
        return self.__points
    
    def perimeter(self):
        perimeter = 0.0
        for i in range(len(self.points)):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % len(self.points)] 
            perimeter += math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
        return perimeter

#Task 2
# Define Vector class for enemy movement
class Vector:
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy
    
    def __repr__(self):
        return f"Vector({self.dx}, {self.dy})"

# Define Enemy class
class Enemy:
    def __init__(self, name, position, vector):
        self.name = name
        self.position = position
        self.vector = vector
        self.life_points = 10
    
    def move(self):
        self.position.x += self.vector.dx
        self.position.y += self.vector.dy
    
    def is_alive(self):
        return self.life_points > 0
    
    def take_damage(self, damage):
        self.life_points -= damage
        if self.life_points <= 0:
            self.life_points = 0  # Ensure non-negative life points
    
    def __repr__(self):
        return f"Enemy({self.name}, {self.position}, {self.life_points} HP)"

# Define Tower class (base class)
class Tower:
    def __init__(self, name, position, attack_points, range):
        self.name = name
        self.position = position
        self.attack_points = attack_points
        self.range = range
    
    def is_in_range(self, enemy):
        dist = math.sqrt((self.position.x - enemy.position.x) ** 2 + (self.position.y - enemy.position.y) ** 2)
        return dist <= self.range
    
    def attack(self, enemy):
        if self.is_in_range(enemy):
            enemy.take_damage(self.attack_points)
    
    def __repr__(self):
        return f"Tower({self.name}, {self.position}, {self.attack_points} AP, {self.range} range)"

# Define BasicTower class (inherits from Tower)
class BasicTower(Tower):
    def __init__(self, name, position):
        super().__init__(name, position, attack_points=1, range=2)

# Define AdvancedTower class (inherits from Tower)
class AdvancedTower(Tower):
    def __init__(self, name, position):
        super().__init__(name, position, attack_points=2, range=4)




# TaskHandler class to handle specific tasks
class TaskHandler:
    @staticmethod
    def run_task_1():
        """
        Executes a series of geometric operations to demonstrate
        parallel/perpendicular checks, area calculation, circle intersection,
        and perimeter calculation.
        """
        # Task 1: Check if two lines are parallel
        lineA = Line(Point(2, 4), Point(-6, 1))
        lineB = Line(Point(2, 2), Point(-6, -1))
        print("Are Line A and Line B parallel?", lineA.is_parallel(lineB))

        # Task 2: Check if two lines are perpendicular
        lineC = Line(Point(-1, 6), Point(-4, -4))
        print("Are Line C and Line A perpendicular?", lineA.is_perpendicular(lineC))

        # Task 3: Calculate circle area
        circleA = Circle(Point(6, 3), 2)
        print(f"Area of Circle A: {circleA.area():.2f}")

        # Task 4: Check if two circles intersect
        circleB = Circle(Point(8, 1), 1)
        print("Do Circle A and Circle B intersect?", circleA.intersects(circleB))

        # Task 5: Calculate perimeter of polygon
        polygonA = Polygon([Point(2, 0), Point(5, -1), Point(4, -4), Point(-1, -2)])
        print(f"Perimeter of Polygon A: {polygonA.perimeter():.2f}")
    
    @staticmethod
    def run_task_2(turns=10):
        # Initialize enemies and towers
        enemies = [
            Enemy("E1", Point(-10, 2), Vector(2, -1)),
            Enemy("E2", Point(-8, 0), Vector(3, 1)),
            Enemy("E3", Point(-9, -1), Vector(3, 0)),
        ]

        towers = [
            BasicTower("T1", Point(-3, 2)),
            BasicTower("T2", Point(-1, -2)),
            BasicTower("T3", Point(4, 2)),
            BasicTower("T4", Point(7, 0)),
            AdvancedTower("A1", Point(1, 1)),
            AdvancedTower("A2", Point(4, -3)),
        ]

        for turn in range(turns):
            #print(f"\n--- Turn {turn + 1} ---")
            
            # Move enemies
            for enemy in enemies:
                if enemy.is_alive():
                    enemy.move()
                    #print(f"{enemy.name} moves to {enemy.position} with {enemy.life_points} HP.")
            
            # Towers attack enemies
            for tower in towers:
                for enemy in enemies:
                    if enemy.is_alive():
                        tower.attack(enemy)
                        if enemy.life_points <= 0:
                            print(f"{enemy.name} is dead!")
            
            # Print status of enemies at the end of the turn
            '''
            for enemy in enemies:
                if enemy.is_alive():
                    print(f"{enemy.name} is at {enemy.position} with {enemy.life_points} HP.")
                else:
                    print(f"{enemy.name} is dead.")
            '''

        # Print the final status of all enemies after 10 turns
        print("\n--- Final Status after 10 turns ---")
        for enemy in enemies:
            print(f"{enemy.name} final position: {enemy.position}, {enemy.life_points} HP")
    
if __name__ == "__main__":
    # Running the task
    TaskHandler.run_task_1()
    TaskHandler.run_task_2()