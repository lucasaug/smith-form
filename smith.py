# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import sys

class PolyReader:   
    def __init__(self, inputFile):
        self.inputFile = inputFile

    def __processEntry(self, entry):
        entry = entry.replace(" ", "")
        entry = entry.replace("-", "+-")
        entry = entry.split("+")

        coefs = []
        for monomial in entry:
            if len(monomial) > 0:
                monomial = monomial.split("x")
                coef = 1
                if len(monomial[0]) != 0:
                    coef = float(monomial[0])

                exponent = 0
                if len(monomial) > 1:
                    if monomial[1]:
                        exponent = int(monomial[1])
                    else:
                        exponent = 1

                coefs.append((coef, exponent))

        coefs.sort(key = lambda x: x[1])
        poly = [0] * (coefs[-1][1] + 1)

        for coef in coefs:
            poly[coef[1]] += coef[0]

        return np.poly1d(poly[::-1])

    def __processLine(self, line):
        line = line.replace("\n", "")
        line = line.split(",")
        nCol = len(line)
        row = []

        for entry in line:
            row.append(self.__processEntry(entry))

        return row, nCol

    def readMatrix(self):
        pols = []
        nCol = None
        with open(self.inputFile, 'r') as fp:
            line = fp.readline().replace("\n","")
            while line:
                row, colsFound = self.__processLine(line)
                pols.append(row)

                if not nCol:
                    nCol = colsFound
                elif nCol != colsFound:
                    raise ValueError("Entrada com quantidade inconsistente de colunas")

                line = fp.readline()
        
        if len(pols) != nCol:
            raise ValueError("O número de linhas e colunas deve ser igual")

        result = np.empty((len(pols), nCol), dtype=object)
        for i in xrange(0, len(pols)):
            for j in xrange(0, nCol):
                result[i, j] = pols[i][j]

        return result


class PolyWriter:
    def printEntry(self, poly):
        strpoly = ""
        counter = poly.order
        coefs = poly.coeffs

        first = True

        for i in xrange(0, poly.order + 1):
            if coefs[i] != 0:
                if not first:
                    strpoly += " "
                    if coefs[i] > 0:
                        strpoly += "+"

                if not (abs(coefs[i]) == 1 and i < poly.order):
                    strpoly += str(coefs[i])
                elif coefs[i] == -1:
                    strpoly += "-"

                if i < poly.order:
                    strpoly += "x"
                    if i < poly.order - 1:
                        strpoly += str(poly.order - i)

                first = False

        if strpoly == "":
            strpoly = "0"

        print(strpoly, end='')

    def printMatrix(self, matrix):
        for i in xrange(0, matrix.shape[0]):
            for j in xrange(0, matrix.shape[1]):
                self.printEntry(matrix[i, j])
                if j < matrix.shape[1] - 1:
                    print(", ", end='')
            print("")


class MatrixSolver:
    def __init__(self, matrix):
        self.matrix = matrix

    def solve(self):
        nRow, nCol = self.matrix.shape
        n = min(nRow, nCol)
        counter = 0

        writer = PolyWriter()

        P = np.empty((nRow, nRow), dtype=object)
        Q = np.empty((nCol, nCol), dtype=object)

        for i in xrange(0, nRow):
            for j in xrange(0, nRow):
                P[i,j] = np.poly1d(0) if i != j else np.poly1d(1)
        for i in xrange(0, nCol):
            for j in xrange(0, nCol):
                Q[i,j] = np.poly1d(0) if i != j else np.poly1d(1)

        while counter < n:
            clearedPositions = False
            while not clearedPositions:
                position = (counter, counter)
                for i in xrange(counter, nRow):
                    for j in xrange(counter, nCol):
                        entry = self.matrix[i, j]
                        if not (entry.order == 0 and np.polyval(entry, 0) == 0):
                            if entry.order < self.matrix[position].order:
                                position = (i, j)

                clearedPositions = True
                for i in xrange(0, nRow):
                    checkPol = matrix[i, counter]
                    if (not (checkPol.order == 0 and np.polyval(checkPol, 0) == 0)) and i != counter:
                        clearedPositions = False
                for i in xrange(0, nRow):
                    checkPol = matrix[counter, i]
                    if (not (checkPol.order == 0 and np.polyval(checkPol, 0) == 0)) and i != counter:
                        clearedPositions = False

                if position == (counter, counter) and clearedPositions:
                    entry = self.matrix[counter, counter]
                    if entry.order == 1 and np.polyval(entry, 0) == 0:
                        counter = n
                    else:
                        if self.matrix[position].coef[0] != 0 and self.matrix[position].coef[0] != 1:
                            print("Divide linha %d por %f" % (counter, self.matrix[position].coef[0]))
                            P[counter, :] /= self.matrix[position].coef[0]
                            self.matrix[position] /= self.matrix[position].coef[0]
                else:
                    if counter != position[0]:
                        print("Troca linhas %d e %d" % (counter + 1, position[0] + 1))
                    if counter != position[1]:
                        print("Troca colunas %d e %d" % (counter + 1, position[1] + 1))

                    # Troca de linhas

                    aux = self.matrix[counter,:].copy()
                    self.matrix[counter,:] = self.matrix[position[0],:].copy()
                    self.matrix[position[0],:] = aux

                    # Matriz P

                    aux = P[counter,:].copy()
                    P[counter,:] = P[position[0],:].copy()
                    P[position[0],:] = aux

                    # Troca de colunas

                    aux = self.matrix[:,counter].copy()
                    self.matrix[:,counter] = self.matrix[:,position[1]].copy()
                    self.matrix[:,position[1]] = aux

                    # Matriz Q

                    aux = Q[:,counter].copy()
                    Q[:,counter] = Q[:,position[1]].copy()
                    Q[:,position[1]] = aux

                    for i in xrange(counter + 1, nRow):
                        q, _ = np.polydiv(self.matrix[i, counter], self.matrix[counter, counter])
                        for j in xrange(0, nCol):
                            self.matrix[i,j] -= q * self.matrix[counter,j]

                            # Matriz P
                            P[i,j] -= q * P[counter, j]

                        if q != np.poly1d(0):
                            print("Subtrai, da linha %d, a linha %d multiplicada por (" % (i+1, counter+1), end='')
                            writer.printEntry(q)
                            print(")")
                    for i in xrange(counter + 1, nCol):
                        q, _ = np.polydiv(self.matrix[counter, i], self.matrix[counter, counter])
                        for j in xrange(0, nRow):
                            self.matrix[j,i] -= q * self.matrix[j,counter]

                            # Matriz Q
                            Q[j,i] -= q * Q[j,counter]

                        if q != np.poly1d(0):
                            print("Subtrai, da coluna %d, a coluna %d multiplicada por (" % (i+1, counter+1), end='')
                            writer.printEntry(q)
                            print(")")

            counter += 1

        return P, self.matrix, Q


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Por favor forneça um arquivo de entrada")
    else:
        inputFile = sys.argv[1]
        reader = PolyReader(inputFile)
        matrix = reader.readMatrix()

        solver = MatrixSolver(matrix)
        print("Operações:")
        print()
        P, S, Q = solver.solve()

        writer = PolyWriter()
        print()
        print("PAQ = S")
        print()
        print("Matriz de operações linha (P)")
        writer.printMatrix(P)
        print()
        print("Matriz de operações coluna (Q)")
        writer.printMatrix(Q)
        print()        
        print("Matriz na forma de Smith (S)")
        writer.printMatrix(S)
