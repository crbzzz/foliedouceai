// Importations nécessaires
import { sortAscending, sortDescending } from './sortingFunctions';

// Fonction de test pour le tri croissant
function testSortAscending() {
    const input = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
    const expectedOutput = [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9];
    const result = sortAscending(input);
    console.log(result);
    console.assert(JSON.stringify(result) === JSON.stringify(expectedOutput), 'Test failed for ascending sort');
}

// Fonction de test pour le tri décroissant
function testSortDescending() {
    const input = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
    const expectedOutput = [9, 6, 5, 5, 5, 4, 3, 3, 2, 1, 1];
    const result = sortDescending(input);
    console.log(result);
    console.assert(JSON.stringify(result) === JSON.stringify(expectedOutput), 'Test failed for descending sort');
}

// Exécution des tests
testSortAscending();
testSortDescending();