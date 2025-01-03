import React from 'react';

const StrategyHistory = ({ items, selectedItem, onSelect }) => {
  const sortedItems = [...items].reverse();

  return (
    <div className="flex flex-col h-full bg-gray-800">
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-lg font-medium text-white">Strategy History</h2>
      </div>
      <div className="flex-grow overflow-y-auto">
        {sortedItems.map((item, index) => (
          <button
            key={index}
            onClick={() => onSelect(item)}
            className={`w-full text-left p-4 hover:bg-gray-700 transition-colors
                      border-b border-gray-700 ${
                        selectedItem === item ? 'bg-gray-700' : ''
                      }`}
          >
            <div className="text-sm font-medium text-white">
              {item.strategy_name}
            </div>
            <div className="text-xs text-gray-400 mt-1">
              {item.strategy_path}
            </div>
          </button>
        ))}
        {items.length === 0 && (
          <div className="p-4 text-sm text-gray-400">
            No strategy history available
          </div>
        )}
      </div>
    </div>
  );
};

export default StrategyHistory; 