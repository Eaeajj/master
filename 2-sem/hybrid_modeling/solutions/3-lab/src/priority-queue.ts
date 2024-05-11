type CompareDir = "asc" | "desc";
export type Comparator<Shape extends {}> = {
  [K in keyof Shape]?: CompareDir;
};

export class PriorityQueue<Model extends {}> {
  queue: Model[];
  comparators: Comparator<Model>[];
  compareFn: (a: Model, b: Model) => number;

  constructor(comparators: Comparator<Model>[], initialArr: Model[] = []) {
    this.comparators = comparators;
    this.compareFn = (a, b) => {
      for (const comparator of comparators) {
        // this should be changed because of string literals comparing
        // but for dates it's ok
        const [fieldName, dir] = Object.entries(comparator)[0];

        if (a[fieldName] > b[fieldName]) return dir === "asc" ? 1 : -1;
        if (a[fieldName] < b[fieldName]) return dir === "asc" ? -1 : 1;
      }
      return 0;
    };

    this.queue = initialArr.toSorted(this.compareFn);
  }

  size() {
    return this.queue.length;
  }

  enqueue(entitity: Model) {
    if (this.queue.length === 0) {
      this.queue.push(entitity);
      return;
    }

    for (let i = 0; i < this.queue.length; i++) {
      const compareResult = this.compareFn(entitity, this.queue[i]);

      if (compareResult !== -1) {
        this.queue.splice(i, 0, entitity);
        return;
      }
    }
    this.queue.push(entitity);
  }

  pop() {
    return this.queue.shift();
  }
}
