import { forwardRef, useState, useRef, useEffect } from "react";
import { thinkingLabel, type ThinkingValue } from "@/utils/thinking";

interface ThinkButtonProps {
  values: ThinkingValue[];
  value: ThinkingValue;
  onChange: (value: ThinkingValue) => void;
  onDropdownToggle?: (isOpen: boolean) => void;
}

export const ThinkButton = forwardRef<HTMLButtonElement, ThinkButtonProps>(
  function ThinkButton({ values, value, onChange, onDropdownToggle }, ref) {
    const isToggle =
      values.length === 2 && values.every((v) => typeof v === "boolean");
    const isFixed = values.length === 1;
    const isVisible = values.length > 0 && !(isFixed && values[0] === false);
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
      if (ref && typeof ref === "object" && ref.current) {
        (ref.current as any).closeDropdown = () => setIsDropdownOpen(false);
      }
    }, [ref, isToggle, isFixed, isVisible]);

    useEffect(() => {
      if (!isDropdownOpen) return;

      function handleClickOutside(event: MouseEvent) {
        if (
          dropdownRef.current &&
          !dropdownRef.current.contains(event.target as Node)
        ) {
          setIsDropdownOpen(false);
        }
      }

      document.addEventListener("mousedown", handleClickOutside);
      return () =>
        document.removeEventListener("mousedown", handleClickOutside);
    }, [isDropdownOpen]);

    if (!isVisible) return null;

    if (isToggle) {
      return (
        <button
          ref={ref}
          type="button"
          aria-label="Thinking"
          aria-pressed={value === true}
          title={value ? "Disable think mode" : "Enable think mode"}
          onClick={() => onChange(!value)}
          className={`select-none flex items-center justify-center rounded-full h-9 w-9 bg-white dark:bg-neutral-700 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer transition-all whitespace-nowrap border border-transparent ${
            value
              ? "text-[rgba(0,115,255,1)] dark:text-[rgba(70,155,255,1)]"
              : "text-neutral-500 dark:text-neutral-400"
          }`}
        >
          <svg
            className="w-3 flex-none fill-current"
            viewBox="0 0 11 19"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path d="M0 4.8125C0 7.8125 1.79688 8.55469 2.29688 13.7656C2.32812 14.0469 2.48438 14.2266 2.78125 14.2266H7.67188C7.97656 14.2266 8.13281 14.0469 8.16406 13.7656C8.66406 8.55469 10.4531 7.8125 10.4531 4.8125C10.4531 2.11719 8.14844 0 5.22656 0C2.30469 0 0 2.11719 0 4.8125ZM1.17969 4.8125C1.17969 2.70312 3.03125 1.17969 5.22656 1.17969C7.42188 1.17969 9.27344 2.70312 9.27344 4.8125C9.27344 7.05469 7.78906 7.58594 7.08594 13.0469H3.375C2.66406 7.58594 1.17969 7.05469 1.17969 4.8125ZM2.75781 15.9141H7.70312C7.96094 15.9141 8.15625 15.7109 8.15625 15.4531C8.15625 15.2031 7.96094 15 7.70312 15H2.75781C2.5 15 2.29688 15.2031 2.29688 15.4531C2.29688 15.7109 2.5 15.9141 2.75781 15.9141ZM5.22656 18.1797C6.4375 18.1797 7.44531 17.5859 7.52344 16.6875H2.9375C2.99219 17.5859 4.00781 18.1797 5.22656 18.1797Z" />
          </svg>
        </button>
      );
    }

    const displayLabel = thinkingLabel(value);
    return (
      <div className="relative" ref={dropdownRef}>
        <button
          ref={ref}
          type="button"
          aria-label={`Thinking: ${displayLabel}`}
          aria-haspopup={isFixed ? undefined : "menu"}
          aria-expanded={isFixed ? undefined : isDropdownOpen}
          disabled={isFixed}
          title={`Thinking: ${displayLabel}${isFixed ? " (fixed by model)" : ""}`}
          onClick={() => {
            if (isFixed) return;
            const newState = !isDropdownOpen;
            setIsDropdownOpen(newState);
            onDropdownToggle?.(newState);
          }}
          className={`select-none flex items-center justify-center gap-1 rounded-full h-9 px-3 bg-white dark:bg-neutral-700 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer transition-all whitespace-nowrap border border-transparent text-[rgba(0,115,255,1)] dark:text-[rgba(70,155,255,1)]`}
        >
          <div className="justify-center items-center flex space-x-2">
            <svg
              className="w-3 flex-none fill-current"
              viewBox="0 0 11 19"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path d="M0 4.8125C0 7.8125 1.79688 8.55469 2.29688 13.7656C2.32812 14.0469 2.48438 14.2266 2.78125 14.2266H7.67188C7.97656 14.2266 8.13281 14.0469 8.16406 13.7656C8.66406 8.55469 10.4531 7.8125 10.4531 4.8125C10.4531 2.11719 8.14844 0 5.22656 0C2.30469 0 0 2.11719 0 4.8125ZM1.17969 4.8125C1.17969 2.70312 3.03125 1.17969 5.22656 1.17969C7.42188 1.17969 9.27344 2.70312 9.27344 4.8125C9.27344 7.05469 7.78906 7.58594 7.08594 13.0469H3.375C2.66406 7.58594 1.17969 7.05469 1.17969 4.8125ZM2.75781 15.9141H7.70312C7.96094 15.9141 8.15625 15.7109 8.15625 15.4531C8.15625 15.2031 7.96094 15 7.70312 15H2.75781C2.5 15 2.29688 15.2031 2.29688 15.4531C2.29688 15.7109 2.5 15.9141 2.75781 15.9141ZM5.22656 18.1797C6.4375 18.1797 7.44531 17.5859 7.52344 16.6875H2.9375C2.99219 17.5859 4.00781 18.1797 5.22656 18.1797Z" />
            </svg>
            <span className="text-sm">{displayLabel}</span>
          </div>
          {!isFixed && (
            <svg
              className={`w-3 h-3`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 9l-7 7-7-7"
              />
            </svg>
          )}
        </button>

        {!isFixed && isDropdownOpen && (
          <div
            role="menu"
            aria-label="Thinking"
            className="absolute bottom-full mb-2 text-[15px] rounded-2xl overflow-hidden bg-white border border-neutral-100 text-neutral-800 shadow-xl shadow-black/5 backdrop-blur-lg dark:border-neutral-600/40 dark:bg-neutral-800 dark:text-white dark:ring-black/20 min-w-[120px]"
          >
            {values.map((level) => (
              <button
                key={`${typeof level}:${level}`}
                type="button"
                role="menuitemradio"
                aria-checked={value === level}
                className={`w-full text-left px-3 py-2 cursor-pointer hover:bg-neutral-100 dark:hover:bg-neutral-700 transition-colors text-neutral-700 dark:text-neutral-300 ${
                  value === level ? "bg-neutral-100 dark:bg-neutral-700/60" : ""
                }`}
                onClick={() => {
                  onChange(level);
                  setIsDropdownOpen(false);
                }}
              >
                {thinkingLabel(level)}
              </button>
            ))}
          </div>
        )}
      </div>
    );
  },
);
