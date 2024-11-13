# WebProtégé Pizza Ontology Tutorial

## Prerequisites
- Access to WebProtégé (https://webprotege.stanford.edu)
- A WebProtégé account

## Step 1: Create a New Project
1. Navigate to WebProtégé
2. Click "Create New Project"
3. Enter project name: "Pizza Ontology"
4. Add description (optional)
5. Click "Create"

## Step 2: Create Base Classes
1. In the "Classes" tab:
   - Click the "Add subclass" button (+ icon)
   - Add these main classes under "owl:Thing":
   ```
   - Pizza
   - PizzaComponent
     └── PizzaBase
     └── PizzaTopping
   ```

## Step 3: Create Pizza Base Types
1. Select "PizzaBase" class
2. Click "Add subclass"
3. Create:
   ```
   PizzaBase
   └── ThinBase
   └── ThickBase
   ```

## Step 4: Create Pizza Topping Hierarchy
1. Select "PizzaTopping" class
2. Add subclasses:
   ```
   PizzaTopping
   └── CheeseTopping
       └── Mozzarella
   └── VegetableTopping
       └── Tomato
   └── MeatTopping
       └── Pepperoni
   ```

## Step 5: Create Object Properties
1. Go to the "Object Properties" tab
2. Click "Add object property" (+ icon)
3. Create two properties:
   ```
   - hasTopping
   - hasBase
   ```

## Step 6: Configure Object Properties
1. For "hasTopping":
   - Click "Domain" (+)
   - Add "Pizza" as domain
   - Click "Range" (+)
   - Add "PizzaTopping" as range

2. For "hasBase":
   - Click "Domain" (+)
   - Add "Pizza" as domain
   - Click "Range" (+)
   - Add "PizzaBase" as range

## Step 7: Create Margherita Pizza Class
1. Return to "Classes" tab
2. Add "MargheritaPizza" as subclass of "Pizza"
3. Add Relationships:
   - Select "MargheritaPizza"
   - Click "Add restriction" 
   - Choose "hasTopping some Mozzarella"
   - Add another Relationships
   - Choose "hasTopping some Tomato"

## Step 8: Add Annotations (Optional)
1. Select any class
2. In the "Annotations" tab
3. Click "+" to add:
   - Labels
   - Comments
   - Descriptions

## Step 9: Save and Verify
1. WebProtégé auto-saves changes
2. Use the "Class hierarchy" view to verify structure
3. Test using the "Query" tab

## Tips & Best Practices
- Regularly download backups of your ontology
- Use the "Discussions" feature for collaboration
- Monitor changes in the "Changes" tab
- Check class usage in the "Usage" tab

## Next Steps
Extend your ontology with:
- Additional pizza types
- More toppings
- New properties (hasSpiciness, hasPrice)
- Disjoint classes
- Complex restrictions

## Common Issues and Solutions
1. **Auto-save not working**: Refresh the page
2. **Cannot create class**: Check permissions
3. **Hierarchy issues**: Use the "Usage" tab to check references
