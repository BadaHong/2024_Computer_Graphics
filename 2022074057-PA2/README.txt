1. Variables
    I included 4 additional variables: clicks, controlPoints, animStartTime, and firstClick.
    - clicks: count the number of clicking on cow. It represents how many control points have been set.
    - controlPoints: an array to store the control points of cow, which is clicking location.
    - animStartTime: represent animation's starting time to help calculating the animation time in display().
    - firstClick: evaluate whether it is the first click on cow. If it is the first click on cow, it just grabs the cow so that the cow can follow the cursor.

2. display()
    For TODO section, I implemented code to consider the number of control points selected so far.
    If it is not yet clicked for six times, it keeps reproduce cows on the control points where exactly the cursor was.
    When six control points are selected, then it calculates the cow's direction and position by using Catmull-Rom spline curve and yaw and pitch orientations to make cow face forward or upward/downward depending on the next position.
    After calculating, the cow is animated following the control points. Once the cow finishes following the track three times, the cow is located in the initial position and direction as shown in the demo video of the professor.

3. Additional Methods
    I created catmullRomSpline() and cowDirection() methods to calculate the cow's position and direction and use them in display() method.
    catmullRomSpline() helps to calculate new position of cow, and cowDirection() helps cow to face the direction of next movement.
    It makes possible to face forward, upward, and downward depending on the next movement.

4. onMouseButton()
    I edited its code to consider the case when the cow is clicked for the first time and when all six control points are selected.
    When the cow is first clicked, it just grabs the cow and makes the cow to move following the cursor.
    Then, when the cow is clicked again, it starts storing control points in controlPoints variable for six times.

5. onMouseDrag()
    I implemented the TODO section by referencing the below code about horizontal dragging.
    Following the below code, I adjusted a dragging plane. It allows the cow to move based on the intersection of a ray from the screen coordinates with a plane perpendicular to the ray direction.

These changes result in producing cow's roller coaster animation exactly the same as the professor's demo video in the lecture video.