#![feature(test)]
//! Type-level safe mutable global access, with support for recursive immutable
//! locking.
//!
//! ```
//! use global::Global;
//!
//! // The global value.
//! static VALUE: Global<i32> = Global::new();
//!
//! // Spawn 100 threads and join them all.
//! let mut threads = Vec::new();
//!
//! for _ in 0..100 {
//!     threads.push(std::thread::spawn(|| {
//!         *VALUE.lock_mut().unwrap() += 1;
//!     }));
//! }
//!
//! for thread in threads {
//!     thread.join().unwrap();
//! }
//!
//! // This value is guaranteed to be 100.
//! assert_eq!(*VALUE.lock().unwrap(), 100);
//! ```

use std::{
    fmt,
    error::Error,
    sync::Arc,
    ops::{Deref, DerefMut},
    cell::{self, RefCell},
    mem::ManuallyDrop,
};

use parking_lot::{Once, ReentrantMutex, ReentrantMutexGuard};

/// A failure occured while borrowing a `Global<T>` value.
///
/// This happens when the value is incorrectly borrowed twice within a single
/// thread. While cross-thread locking will simply block until the value is
/// available, within a single thread potential deadlocks are detected. An error
/// can occur in one of three ways:
///
/// * Attempting to immutably borrow a value that is already mutably borrowed.
/// * Attempting to mutably borrow a value that is already mutably borrowed.
/// * Attempting to mutably borrow a value that is already immutably borrowed.
///
/// Note that this error will not be raised in the case of multiple immutable
/// borrows.
#[derive(Debug)]
pub struct BorrowFail;

impl fmt::Display for BorrowFail {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "failed to borrow global value twice in same thread")
    }
}

impl Error for BorrowFail {}

/// A value slot stored inside of `Global<T>`.
type InnerPointer<T> = Arc<ReentrantMutex<RefCell<T>>>;

/// A mutable global value.
///
/// All types wrapped in `Global` must implement [`Default`]. This gives the
/// initial value of the global variable.
///
/// Handles to this value can be obtained with the `Global::lock` or
/// `Global::lock_mut` methods.
///
/// [`Default`]: https://doc.rust-lang.org/std/default/trait.Default.html
pub struct Global<T>(Immutable<InnerPointer<T>>);

// The inner value is only used to make an immutable call to `.clone()`. The
// only time it is mutated is within the `Once` guard. This means all threads
// will attempt to get *immutable* access and block until only one thread as
// succeeded. That makes this `impl` safe only if `.ensure_exists()` is called
// whenever accessing the inner `UnsafeCell` value.
//
// This bound is on `T: Send` as `Mutex<T>` requires it to implement `Sync`.
// Because the mutex is in a static position it must be sync, so we need to
// ensure this bound is satisfied.
unsafe impl<T> Sync for Global<T> where T: Send {}

impl<T> Global<T> {
    /// Construct a new instance of self.
    pub const fn new() -> Self {
        Self(Immutable::new())
    }
}

impl<T: Default + Send + 'static> Global<T> {
    /// Run a closure on an immutable reference to the inner value.
    ///
    /// This will return the closure's return type. Internally, [`lock`] is
    /// called and unwrapped.
    ///
    /// [`lock`]: #method.lock
    ///
    /// # Panics
    ///
    /// This will panic if the value is already borrowed mutably in the same
    /// thread.
    pub fn with<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&T) -> R,
    {
        f(&*self.lock().expect("Couldn't immutably access global variable"))
    }

    /// Run a closure on a mutable reference to the inner value.
    ///
    /// This will return the closure's return type. Internally, [`lock_mut`] is
    /// called and unwrapped.
    ///
    /// [`lock_mut`]: #method.lock_mut
    ///
    /// # Panics
    ///
    /// This will panic if the value is already borrowed either mutably or
    /// immutably in the same thread.
    pub fn with_mut<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        f(&mut *self.lock_mut().expect("Couldn't mutably access global variable"))
    }

    /// Obtain a lock providing an immutable reference to the inner value0.
    ///
    /// This method will block the current thread until any other threads
    /// holding a lock are destroyed. If the current thread already has mutable
    /// access, this method will return an error.
    pub fn lock(&self) -> Result<GlobalGuard<T>, BorrowFail> {
        // The borrow checker is not able to prove what we are doing here is
        // safe. Essentially, the global value lives on the heap and is
        // guaranteed not to move, so we can treat the inner value as if it is a
        // 'static reference as long as we make sure to tear down the guard
        // correctly. For this to work, pointers are used to erase lifetime data
        // and then a `Drop` implementation manually drops the fields in the
        // correct order.

        let mutex: Arc<_> = Arc::clone(&*self.0);
        let mutex_ptr = &*mutex as *const ReentrantMutex<RefCell<T>>;

        let mutex_guard = unsafe { (*mutex_ptr).lock() };
        let mutex_guard_ptr = &*mutex_guard as *const RefCell<T>;

        let ref_cell_guard = unsafe {
            (*mutex_guard_ptr)
                .try_borrow()
                .map_err(|_| BorrowFail)?
        };

        Ok(GlobalGuard {
            mutex: ManuallyDrop::new(mutex),
            mutex_guard: ManuallyDrop::new(mutex_guard),
            ref_cell_guard: ManuallyDrop::new(ref_cell_guard),
        })
    }

    /// Obtain a lock providing a mutable reference to the inner value0.
    ///
    /// This method will block the current thread until any other threads
    /// holding a lock are destroyed. If the current thread already has access,
    /// whether mutable or immutable, this method will return an error.
    pub fn lock_mut(&self) -> Result<GlobalGuardMut<T>, BorrowFail> {
        // The body here is largely the same as `lock`. Comments there explain
        // what is going on.

        let mutex: Arc<_> = Arc::clone(&*self.0);
        let mutex_ptr = &*mutex as *const ReentrantMutex<RefCell<T>>;

        let mutex_guard = unsafe { (*mutex_ptr).lock() };
        let mutex_guard_ptr = &*mutex_guard as *const RefCell<T>;

        let ref_cell_guard = unsafe {
            (*mutex_guard_ptr)
                .try_borrow_mut()
                .map_err(|_| BorrowFail)?
        };

        Ok(GlobalGuardMut {
            mutex: ManuallyDrop::new(mutex),
            mutex_guard: ManuallyDrop::new(mutex_guard),
            ref_cell_guard: ManuallyDrop::new(ref_cell_guard),
        })
    }
}

/// A mutable handle to some `Global<T>` value.
///
/// This type implements `Deref<Target = T>` and `DerefMut`. Once this guard is
/// dropped, the global this guard is locking is immediately unlocked.
///
/// ```
/// # use global::Global;
/// static VALUE: Global<i32> = Global::new();
///
/// let mut guard = VALUE.lock_mut().unwrap();
///
/// *guard += 1;
/// *guard += 1;
/// *guard += 1;
///
/// assert_eq!(*guard, 3);
/// ```
pub struct GlobalGuardMut<T: 'static> {
    // These are marked manually drop to specify drop order. In a perfect world,
    // the guard would bear the lifetime of the mutex, however that requires
    // rust to have self-referential structs, which it currently does not have.
    mutex: ManuallyDrop<Arc<ReentrantMutex<RefCell<T>>>>,
    mutex_guard: ManuallyDrop<ReentrantMutexGuard<'static, RefCell<T>>>,
    ref_cell_guard: ManuallyDrop<cell::RefMut<'static, T>>,
}

impl<T: 'static> Drop for GlobalGuardMut<T> {
    fn drop(&mut self) {
        // Drop the guard *before* the mutex.
        unsafe {
            ManuallyDrop::drop(&mut self.ref_cell_guard);
            ManuallyDrop::drop(&mut self.mutex_guard);
            ManuallyDrop::drop(&mut self.mutex);
        }
    }
}

impl<T: 'static> Deref for GlobalGuardMut<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &*self.ref_cell_guard
    }
}

impl<T: 'static> DerefMut for GlobalGuardMut<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut *self.ref_cell_guard
    }
}

/// An immutable handle to some `Global<T>` value.
///
/// This type implements `Deref<Target = T>`. Once this guard is dropped, the
/// global this guard is locking is immediately unlocked.
///
/// ```
/// # use global::Global;
/// static VALUE: Global<i32> = Global::new();
///
/// assert_eq!(0, *VALUE.lock().unwrap());
/// ```
pub struct GlobalGuard<T: 'static> {
    // These are marked manually drop to specify drop order. In a perfect world,
    // the guard would bear the lifetime of the mutex, however that requires
    // rust to have self-referential structs, which it currently does not have.
    mutex: ManuallyDrop<Arc<ReentrantMutex<RefCell<T>>>>,
    mutex_guard: ManuallyDrop<ReentrantMutexGuard<'static, RefCell<T>>>,
    ref_cell_guard: ManuallyDrop<cell::Ref<'static, T>>,
}

impl<T: 'static> Drop for GlobalGuard<T> {
    fn drop(&mut self) {
        // Drop the guard *before* the mutex.
        unsafe {
            ManuallyDrop::drop(&mut self.ref_cell_guard);
            ManuallyDrop::drop(&mut self.mutex_guard);
            ManuallyDrop::drop(&mut self.mutex);
        }
    }
}

impl<T: 'static> Deref for GlobalGuard<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &*self.ref_cell_guard
    }
}

/// An immutable global value.
///
/// All types wrapped in `Immutable` must implement [`Default`]. This gives the
/// initial value of the global variable.
///
/// This type can be directly dereferenced. Initialization occurs upon the
/// first dereference.
///
/// [`Default`]: https://doc.rust-lang.org/std/default/trait.Default.html
pub struct Immutable<T> {
    once: Once,
    inner: std::mem::MaybeUninit<T>,
}

unsafe impl<T: Send> Send for Immutable<T> {}
unsafe impl<T: Sync> Sync for Immutable<T> {}

impl<T: Default> Immutable<T> {
    /// Ensure the inner value exists.
    ///
    /// This method *must* be called before accessing the inner `UnsafeCell`.
    fn ensure_exists(&self) {
        self.once.call_once(|| {
            // This is safe as this assignment can only be called once, hence no
            // hint of race conditions. Other threads will be blocked until this
            // is done.
            unsafe {
                (self.inner.as_ptr() as *mut T).write(T::default());
            }
        });
    }
}

impl<T> Immutable<T> {
    /// The initial global value.
    pub const fn new() -> Self {
        Self {
            once: Once::new(),
            inner: std::mem::MaybeUninit::uninit()
        }
    }
}

impl<T: Default> Deref for Immutable<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // Unwrap cannot panic, we called `ensure_exists`.
        self.ensure_exists();
        unsafe {
            &*self.inner.as_ptr()
        }
    }
}

#[cfg(test)]
mod test {
    use std::{
        thread,
        sync::mpsc,
        time::Duration,
    };

    use super::{Immutable, Global};
    extern crate test;

    #[test]
    fn no_race_condition() {
        static NUM: Global<i32> = Global::new();

        let mut v = Vec::new();

        for _ in 0..1000 {
            v.push(thread::spawn(|| {
                for _ in 0..100 {
                    *NUM.lock_mut().unwrap() += 1;
                }
            }));
        }

        for thread in v {
            thread.join().unwrap();
        }

        assert_eq!(*NUM.lock().unwrap(), 100_000);
    }

    // Ensure a lock will block.
    #[test]
    fn no_race_extended_lock() {
        static NUM: Global<i32> = Global::new();

        let (tx, rx) = mpsc::channel();

        let t1 = thread::spawn(move || {
            let mut lock = NUM.lock_mut().unwrap();

            // Go.
            tx.send(()).unwrap();

            thread::sleep(Duration::new(0, 1_000_000));

            *lock += 1;
        });

        let t2 = thread::spawn(move || {
            // Wait for the signal.
            let () = rx.recv().unwrap();

            *NUM.lock_mut().unwrap() += 1;
        });

        t1.join().unwrap();
        t2.join().unwrap();

        assert_eq!(*NUM.lock().unwrap(), 2);
    }

    #[test]
    #[should_panic]
    fn borrow_immutably_while_mutably_borrowed() {
        static NUM: Global<i32> = Global::new();

        let _x = NUM.lock_mut().unwrap();
        let _y = NUM.lock().unwrap();
    }

    #[test]
    #[should_panic]
    fn borrow_mutably_while_mutably_borrowed() {
        static NUM: Global<i32> = Global::new();

        let _x = NUM.lock_mut().unwrap();
        let _y = NUM.lock_mut().unwrap();
    }

    #[test]
    #[should_panic]
    fn borrow_mutably_while_immutably_borrowed() {
        static NUM: Global<i32> = Global::new();

        let _x = NUM.lock().unwrap();
        let _y = NUM.lock_mut().unwrap();
    }

    #[test]
    fn borrow_immutably_while_immutably_borrowed() {
        static NUM: Global<i32> = Global::new();

        let _x = NUM.lock().unwrap();
        let _y = NUM.lock().unwrap();
    }

    /// Test recursive immutable locking with two interleaved threads.
    #[test]
    fn complex_thread_interactions() {
        static NUM: Global<i32> = Global::new();

        let lock1 = NUM.lock().unwrap();
        let lock2 = NUM.lock().unwrap();
        let lock3 = NUM.lock().unwrap();

        let t = thread::spawn(|| {
            *NUM.lock_mut().unwrap() += 1;

            assert!(NUM.lock().is_ok());
        });

        thread::sleep(Duration::from_millis(100));

        assert!(NUM.lock_mut().is_err());

        drop(lock1);
        drop(lock2);
        drop(lock3);

        *NUM.lock_mut().unwrap() += 1;

        t.join().unwrap();

        assert_eq!(2, *NUM.lock().unwrap());
    }

    #[bench]
    fn benchmark_immutable(b: &mut test::Bencher) {
        b.iter(|| {
            let n = test::black_box(1_000_000);
            (0..n).for_each(|_| {
                static N: Immutable<i32> = Immutable::new();
                &*N;
            });

        });
    }
}
