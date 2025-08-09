// Fonction de tri croissant/décroissant en TypeScript
function triCroissantDecroissant(tableau: number[], ordre: 'asc' | 'desc'): number[] {
    if (ordre === 'asc') {
        return tableau.sort((a, b) => a - b);
    } else if (ordre === 'desc') {
        return tableau.sort((a, b) => b - a);
    } else {
        throw new Error("Ordre invalide");
    }
}

// Tests avec pytest
import { test, expect } from 'vitest';

test('triCroissantDecroissant - tri croissant', () => {
    const resultat = triCroissantDecroissant([5, 3, 8, 4, 2], 'asc');
    expect(resultat).toEqual([2, 3, 4, 5, 8]);
});

test('triCroissantDecroissant - tri décroissant', () => {
    const resultat = triCroissantDecroissant([5, 3, 8, 4, 2], 'desc');
    expect(resultat).toEqual([8, 5, 4, 3, 2]);
});

test('triCroissantDecroissant - ordre invalide', () => {
    expect(() => triCroissantDecroissant([5, 3, 8, 4, 2], 'invalid' as 'asc' | 'desc')).toThrow('Ordre invalide');
});